"""Command line interface."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('-n', '--nexp', type=int, help='Number of exposures')
    p.add_argument('--max-seconds', type=int, default=300,
                   help='Time limit for solver')
    p.add_argument('--output', '-o', metavar='OUTPUT.ecsv',
                   type=FileType('w'), default='-',
                   help='output filename')
    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    # import shlex
    import sys

    from astropy_healpix import nside_to_level
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from pyscipopt import Model, quicksum
    from scipy.signal import convolve
    from tqdm import tqdm

    from .. import orbit
    from ..regard import get_field_of_regard
    from .. import skygrid

    log.info('reading sky map')

    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, nside_to_level(skygrid.healpix.nside))['PROB']
    if skygrid.healpix.order == 'ring':
        prob = prob[skygrid.healpix.ring_to_nested(np.arange(len(prob)))]

    times = np.arange(orbit.time_steps) * orbit.time_step_duration + start_time

    log.info('generating model')
    m = Model()

    log.info('adding variable: observing schedule')
    shape = (len(skygrid.centers), len(skygrid.rolls),
             orbit.time_steps - orbit.time_steps_per_exposure + 1)
    schedule = np.reshape(
        [m.addVar(vtype='B') for _ in tqdm(range(np.prod(shape)))], shape)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = np.asarray(
        [m.addVar(vtype='B') for _ in tqdm(range(skygrid.healpix.npix))])

    log.info('adding variable: whether a given field is used')
    field_used = np.reshape(
        [m.addVar(vtype='B') for _ in tqdm(range(np.prod(shape[:2])))],
        shape[:2])

    log.info('adding variable: whether a given time step is used')
    time_used = np.asarray(
        [m.addVar(vtype='B') for _ in tqdm(range(shape[2]))])

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.addCons(time_used.sum() <= 0)

    log.info('adding constraint: only observe one field at a time')
    for i in tqdm(range(shape[2])):
        m.addCons(quicksum(schedule[..., i].ravel()) == time_used[i])
        m.addCons(quicksum(time_used[i:i+orbit.time_steps_per_exposure]) <= 1)

    log.info('adding constraint: a pixel is observed if it is in any field')
    for lhs, rhs in zip(
            tqdm(schedule.reshape(field_used.size, -1)),
            field_used.ravel()):
        m.addCons(quicksum(lhs) >= rhs)
    indices = [[] for _ in range(skygrid.healpix.npix)]
    with tqdm(total=len(skygrid.centers) * len(skygrid.rolls)) as progress:
        for i, grid_i in enumerate(skygrid.get_footprint_grid()):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    indices[k].append((i, j))
                progress.update()
    for lhs_indices, rhs in zip(tqdm(indices), pixel_observed):
        m.addCons(quicksum(field_used[inds] for inds in lhs_indices) >= rhs)

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(
            ~get_field_of_regard(times),
            np.ones(orbit.time_steps_per_exposure)[:, np.newaxis],
            mode='valid', method='direct'))
    for _ in schedule[j, :, i].ravel():
        m.fixVar(_, 0)
    # m.addCons(quicksum(schedule[j, :, i].ravel()) <= 0)

    log.info('adding objective')
    m.setObjective(quicksum(prob * pixel_observed), 'maximize')

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    m.optimize()
    stopwatch.stop()

    log.info('extracting results')
    schedule_flags = np.asarray(
        [m.getVal(_) for _ in schedule.ravel()], dtype=bool
    ).reshape(
        schedule.shape
    )

    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        {
            'time': times[itime],
            'center': skygrid.centers[ipix],
            'roll': skygrid.rolls[iroll]
        }, meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': m.getObjVal(),
            'status': m.getStatus(),
            'real': stopwatch.real,
            'user': stopwatch.user,
            'sys': stopwatch.sys
        }
    )
    result.sort('time')
    result.write(args.output, format='ascii.ecsv')

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
