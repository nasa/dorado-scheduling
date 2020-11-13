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
    from gurobipy import GRB, Model, quicksum
    import numpy as np
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
    schedule = m.addMVar(shape, vtype=GRB.BINARY)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.addMVar(int(skygrid.healpix.npix), vtype=GRB.BINARY)

    log.info('adding variable: whether a given field is used')
    field_used = m.addMVar(shape[:2], vtype=GRB.BINARY)

    log.info('adding variable: whether a given time step is used')
    time_used = m.addMVar(shape[2], vtype=GRB.BINARY)

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.addConstr(time_used.sum() <= 0)

    log.info('adding constraint: only observe one field at a time')
    m.addConstrs(
        schedule[..., i].sum() == time_used[i]
        for i in tqdm(range(shape[2]))
    )
    m.addConstrs(
        time_used[i:i+orbit.time_steps_per_exposure].sum() <= 1
        for i in tqdm(range(shape[2]))
    )

    log.info('adding constraint: a pixel is observed if it is in any field')
    m.addConstrs(schedule[i, j].sum() >= field_used[i, j] for i in tqdm(range(schedule.shape[0])) for j in range(schedule.shape[1]))
    exprs = [0] * skygrid.healpix.npix
    with tqdm(total=len(skygrid.centers) * len(skygrid.rolls)) as progress:
        for i, grid_i in enumerate(skygrid.get_footprint_grid()):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    exprs[k] += field_used[i, j]
                progress.update()
    m.addConstrs(exprs[i] >= pixel_observed[i] for i in tqdm(range(len(exprs))))

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(
            ~get_field_of_regard(times),
            np.ones(orbit.time_steps_per_exposure)[:, np.newaxis],
            mode='valid', method='direct'))
    m.addConstr(schedule[j, :, i].sum() <= 0)

    log.info('adding objective')
    m.setObjective(prob @ pixel_observed, GRB.MAXIMIZE)

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    m.optimize()
    stopwatch.stop()

    log.info('extracting results')
    if m.Status in {GRB.Status.OPTIMAL,
                    GRB.Status.SUBOPTIMAL}:
        status = 'OPTIMAL' if m.Status == GRB.Status.OPTIMAL else 'FEASIBLE'
        schedule_flags = schedule.x.astype(bool)
        objective_value = m.ObjVal
    else:
        schedule_flags = np.zeros(schedule.shape, dtype=bool)
        objective_value = 0.0
        status = 'INFEASIBLE'

    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        {
            'time': times[itime],
            'center': skygrid.centers[ipix],
            'roll': skygrid.rolls[iroll]
        }, meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': objective_value,
            'status': status,
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
