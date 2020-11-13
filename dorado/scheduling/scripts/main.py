"""Command line interface."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('-n', '--nexp', type=int, help='Number of exposures')
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
    from scipy.signal import convolve
    from tqdm import tqdm
    import xpress as xp

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

    m = []

    log.info('adding variable: observing schedule')
    shape = (len(skygrid.centers), len(skygrid.rolls),
             orbit.time_steps - orbit.time_steps_per_exposure + 1)
    schedule = xp.vars(*shape, name='s', vartype=xp.binary)
    m.append(schedule)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = xp.vars(skygrid.healpix.npix, name='p', vartype=xp.binary)
    m.append(pixel_observed)

    log.info('adding variable: whether a given field is used')
    field_used = xp.vars(*shape[:2], name='f', vartype=xp.binary)
    m.append(field_used)

    log.info('adding variable: whether a given time step is used')
    time_used = xp.vars(shape[2], name='t', vartype=xp.binary)
    m.append(time_used)

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.append(xp.Sum(time_used) <= 0)

    log.info('adding constraint: only observe one field at a time')
    for i in tqdm(range(shape[2])):
        m.append(xp.Sum(schedule[..., i].ravel()) == time_used[i])
        m.append(xp.Sum(time_used[i:i+orbit.time_steps_per_exposure]) <= 1)

    log.info('adding constraint: a pixel is observed if it is in any field')
    for lhs, rhs in zip(
            tqdm(schedule.reshape(field_used.size, -1)),
            field_used.ravel()):
        m.append(xp.Sum(lhs) >= rhs)
    indices = [[] for _ in range(skygrid.healpix.npix)]
    with tqdm(total=len(skygrid.centers) * len(skygrid.rolls)) as progress:
        for i, grid_i in enumerate(skygrid.get_footprint_grid()):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    indices[k].append((i, j))
                progress.update()
    for lhs_indices, rhs in zip(tqdm(indices), pixel_observed):
        m.append(xp.Sum(field_used[inds] for inds in lhs_indices) >= rhs)

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(
            ~get_field_of_regard(times),
            np.ones(orbit.time_steps_per_exposure)[:, np.newaxis],
            mode='valid', method='direct'))
    m.append(xp.Sum(schedule[j, :, i].ravel()) <= 0)

    log.info('adding objective')
    m.append(xp.Sum(prob * pixel_observed))

    log.info('creating problem')
    problem = xp.problem(*m, sense=xp.maximize)

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    problem.solve()
    stopwatch.stop()

    log.info('extracting results')
    schedule_flags = problem.getSolution(schedule.astype(object)).astype(bool)
    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        {
            'time': times[itime],
            'center': skygrid.centers[ipix],
            'roll': skygrid.rolls[iroll]
        }, meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': problem.getObjVal(),
            'status': problem.getProbStatusString(),
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
