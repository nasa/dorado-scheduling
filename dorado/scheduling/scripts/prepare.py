"""Generate initial MIP problem representation."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('output', type=FileType('wb'), metavar='FILE.lp.zst',
                   help='Name for zstandard-compressed output file')
    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    import tempfile

    import mip
    from tqdm import tqdm
    from zstandard import ZstdCompressor

    from .. import orbit
    from .. import skygrid

    m = mip.Model()

    log.info('adding variable: observing schedule')
    schedule = m.add_var_tensor(
        (len(skygrid.centers), len(skygrid.rolls),
         orbit.time_steps - orbit.time_steps_per_exposure + 1),
        's', var_type=mip.BINARY)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.add_var_tensor(
        (skygrid.healpix.npix,), 'p', var_type=mip.BINARY)

    log.info('adding constraint: number of exposures')
    m += mip.xsum(schedule.ravel()) <= 0, 'nexp'

    log.info('adding constraint: only observe one field at a time')
    for i in tqdm(range(schedule.shape[-1])):
        m += mip.xsum(
            schedule[..., i:i+orbit.time_steps_per_exposure].ravel()) <= 1

    log.info('adding constraint: a pixel is observed if it is in any field')
    exprs = [-observed >= 0 for observed in pixel_observed]
    field_observed = [[mip.xsum(__) for __ in _] for _ in tqdm(schedule)]
    with tqdm(total=len(skygrid.centers) * len(skygrid.rolls)) as progress:
        for i, grid_i in enumerate(skygrid.get_footprint_grid()):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    exprs[k].add_expr(field_observed[i][j])
                progress.update()
    del field_observed
    for expr in tqdm(exprs):
        m += expr
    del exprs

    log.info('writing output')
    with tempfile.NamedTemporaryFile(suffix='.lp') as uncompressed:
        m.write(uncompressed.name)
        ZstdCompressor().copy_stream(uncompressed, args.output)

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
