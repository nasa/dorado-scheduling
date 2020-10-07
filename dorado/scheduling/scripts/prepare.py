"""Generate initial MIP problem representation."""
import logging
import tempfile

from ligo.skymap.tool import ArgumentParser, FileType
import mip
from zstandard import ZstdCompressor

from .. import orbit
from .. import skygrid

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('output', type=FileType('wb'), metavar='FILE.lp.zst',
                   help='Name for zstandard-compressed output file')
    return p


def main(args=None):
    args = parser().parse_args(args)
    if not args.output.name.endswith('.lp.zst'):
        raise ValueError('The output file extension must be .lp.zst.')

    m = mip.Model()

    log.info('adding variable: observing schedule')
    schedule = m.add_var_tensor(
        (orbit.exposures_per_orbit, len(skygrid.centers), len(skygrid.rolls)),
        'sched', var_type=mip.BINARY)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.add_var_tensor(
        (skygrid.healpix.npix,), 'pix', var_type=mip.BINARY)

    log.info('adding constraint: only observe one field at a time')
    for slice in schedule:
        m += mip.xsum(slice.ravel()) <= 1

    log.info('adding constraint: a pixel is observed if it is contained in any field')
    exprs = [-observed >= 0 for observed in pixel_observed]
    for i, grid_i in enumerate(skygrid.get_footprint_grid()):
        for j, grid_ij in enumerate(grid_i):
            for k in grid_ij:
                exprs[k].add_expr(mip.xsum(schedule[:, i, j]))
    for expr in exprs:
        m.add_constr(expr)

    log.info('writing output')
    with tempfile.NamedTemporaryFile(suffix='.lp') as uncompressed:
        m.write(uncompressed.name)
        ZstdCompressor().copy_stream(uncompressed, args.output)


if __name__ == '__main__':
    main()
