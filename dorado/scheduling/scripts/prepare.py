"""Generate initial MIP problem representation."""
import logging
import tempfile

from ligo.skymap.tool import ArgumentParser, FileType
import mip
from zstandard import ZstdCompressor

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

    log.info('adding variable: whether a given field is observed')
    field_observed = m.add_var_tensor(
        (len(skygrid.centers), len(skygrid.rolls)),
        'field_observed', var_type=mip.BINARY)

    log.info('adding variable: number of fields observed')
    num_fields = m.add_var('num_fields', var_type=mip.INTEGER, lb=0)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.add_var_tensor(
        (skygrid.healpix.npix,), 'pixel_observed', var_type=mip.BINARY)

    log.info('adding constraint: pixels in fields observed')
    exprs = [-observed >= 0 for observed in pixel_observed]
    for i, grid_i in enumerate(skygrid.get_footprint_grid()):
        for j, grid_ij in enumerate(grid_i):
            for k in grid_ij:
                exprs[k].add_var(field_observed[i, j])
    for expr in exprs:
        m.add_constr(expr)

    log.info('adding contstraint: number of fields observed')
    m.add_constr(mip.xsum(field_observed.ravel()) == num_fields)

    log.info('writing output')
    with tempfile.NamedTemporaryFile(suffix='.lp') as uncompressed:
        m.write(uncompressed.name)
        ZstdCompressor().copy_stream(uncompressed, args.output)


if __name__ == '__main__':
    main()
