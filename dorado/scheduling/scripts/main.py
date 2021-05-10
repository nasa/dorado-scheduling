#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Generate an observing plan for a HEALPix probability map."""
import logging

from astropy import units as u
from ligo.skymap.tool import ArgumentParser, FileType
import numpy as np

from .. import mission as _mission
from .. import skygrid
from ..units import equivalencies

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling')

    group = p.add_argument_group(
        'problem setup options',
        'Options that control the problem setup')
    group.add_argument(
        'skymap', metavar='FILE.fits[.gz]',
        type=FileType('rb'), help='Input sky map')
    group.add_argument(
        '-n', '--nexp', type=int, help='Number of exposures')
    group.add_argument(
        '--mission', choices=set(_mission.__all__) - {'Mission'},
        default='dorado', help='Mission configuration')
    group.add_argument(
        '--exptime', type=u.Quantity, default='10 min',
        help='Exposure time (any time units)')
    group.add_argument(
        '--delay', type=u.Quantity, default='30 min',
        help='Delay after event time to start observing (any time units)')
    group.add_argument(
        '--duration', type=u.Quantity, default='1 orbit',
        help='Duration of observing plan (any time units)')

    group = p.add_argument_group(
        'discretization options',
        'Options that control the discretization of decision variables')
    group.add_argument(
        '--time-step', type=u.Quantity, default='1 min',
        help='Model time step (any time units)')
    group.add_argument(
        '--roll-step', type=u.Quantity, default='10 deg',
        help='Roll angle step (any angle units)')
    group.add_argument(
        '--skygrid-step', type=u.Quantity, default='0.0011 sr',
        help='Sky grid resolution (any solid angle units')

    group = group.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--skygrid-method', default='healpix',
        choices=[key.replace('_', '-') for key in skygrid.__all__],
        help='Sky grid method')
    group.add_argument(
        '--skygrid-file', metavar='TILES.ecsv',
        type=FileType('rb'),
        help='tiles filename')

    p.add_argument(
        '--nside', type=int, default=32, help='HEALPix sampling resolution')
    p.add_argument(
        '--timeout', type=int, help='Impose timeout on solutions')
    p.add_argument(
        '--output', '-o', metavar='OUTPUT.ecsv', type=FileType('w'),
        default='-', help='output filename')
    p.add_argument(
        '-j', '--jobs', type=int, default=1, const=None, nargs='?',
        help='Number of threads')
    return p


def nonzero_intervals(a):
    """Find the intervals over which an array is nonzero.

    Examples
    --------
    >>> nonzero_intervals([])
    array([], shape=(0, 2), dtype=int64)
    >>> nonzero_intervals([0, 0, 0, 0])
    array([], shape=(0, 2), dtype=int64)
    >>> nonzero_intervals([1, 1, 1, 1])
    array([[0, 3]])
    >>> nonzero_intervals([0, 1, 1, 1])
    array([[1, 3]])
    >>> nonzero_intervals([1, 1, 1, 0])
    array([[0, 2]])
    >>> nonzero_intervals([1, 1, 0, 1, 0, 1, 1, 1])
    array([[0, 1],
           [3, 3],
           [5, 7]])
    """
    a = np.pad(np.asarray(a, dtype=bool), 1)
    return np.column_stack((np.flatnonzero(a[1:-1] & ~a[:-2]),
                            np.flatnonzero(a[1:-1] & ~a[2:])))


def slew_time(x, v, a):
    """Calculate the time to execute an optimal slew of a given distance.

    The optimal slew consists of an acceleration phase at the maximum
    acceleration, possibly a coasting phase at the maximum angular velocity,
    and a deceleration phase at the maximum acceleration.

    Parameters
    ----------
    x : float, numpy.ndarray
        Distance.
    v : float, numpy.ndarray
        Maximum velocity.
    a : float, numpy.ndarray
        Maximum acceleration.
    """
    xc = np.square(v) / a
    return np.where(x <= xc, np.sqrt(4 * x / a), (x + xc) / v)


# FIXME: this doesn't handle slews with different roll angles yet.
# We probably want to start representing the pointing of the telescope using
# quaternions, which among other things would make it easy to calculate the
# angle between two different pointings with different roll angles.
def slew_time_matrix(centers, v, a):
    return slew_time(centers[:, np.newaxis].separation(centers), v, a)


# FIXME: This should go in the dorado.scheduling.mission.Mission classes.
# The Dorado max velocity is less than 1 deg/s, but slow it down so it is
# obvious whether or not the slew constraints are working.
VELOCITY = 0.1 * u.deg / u.s
"""Max angular velocity"""

# FIXME: This should go in the dorado.scheduling.mission.Mission classes.
ACCELERATION = 0.25 * u.deg / u.s**2
"""Max angular rate"""


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    # import shlex
    import sys

    from astropy_healpix import HEALPix
    from astropy.coordinates import ICRS
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    # FIXME: license check fails without this line
    # (this issue is specific to docplex.cp, but not docplex.mp)
    import cplex  # noqa: F401
    from docplex.cp import model as cp
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from tqdm import tqdm

    mission = getattr(_mission, args.mission)
    healpix = HEALPix(args.nside, order='nested', frame=ICRS())

    log.info('reading sky map')
    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, healpix.level)['PROB']

    # Set up grids
    with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
        time_steps_per_exposure = int(np.round(
            (args.exptime / args.time_step).to_value(
                u.dimensionless_unscaled)))
        times = start_time + args.delay + np.arange(
            0, args.duration.to_value(u.s),
            args.time_step.to_value(u.s)) * u.s
    rolls = np.arange(0, 90, args.roll_step.to_value(u.deg)) * u.deg

    # FIXME: not handling slews between fields of different rolls here...
    if len(rolls) != 1:
        raise NotImplementedError(
            'Slews changing roll angles are not yet implemented')

    if args.skygrid_file is not None:
        centers = Table.read(args.skygrid_file, format='ascii.ecsv')['center']
    else:
        centers = getattr(skygrid, args.skygrid_method.replace('-', '_'))(
            args.skygrid_step)

    log.info('evaluating field of regard')
    regard = mission.get_field_of_regard(centers, times, jobs=args.jobs)

    log.info('generating model')
    m = cp.CpoModel()

    log.info('adding variable: observing schedule')
    schedule = [m.interval_var_list(len(rolls), length=time_steps_per_exposure,
                                    start=[0, len(times)], end=[0, len(times)],
                                    optional=True)
                for _ in range(len(centers))]
    schedule_flat = sum(schedule, [])
    schedule_presence = [[cp.presence_of(_) for _ in __] for __ in schedule]
    schedule_presence_flat = sum(schedule_presence, [])

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.binary_var_list(healpix.npix)

    log.info('adding constraint: slew time')
    # FIXME: not handling orbital period time units conversion here...
    slew_times = np.round(
        (
            slew_time_matrix(centers, VELOCITY, ACCELERATION) / args.time_step
        ).to_value(u.dimensionless_unscaled)
    ).astype(int)
    m.add(cp.no_overlap(
        cp.sequence_var(schedule_flat), cp.transition_matrix(slew_times)))

    log.info('adding constraint: field of regard')
    for schedule_var, not_regard in zip(schedule_flat, ~regard.T):
        forbidden = nonzero_intervals(not_regard)
        if len(forbidden) > 0:
            m.add(cp.no_overlap([
                schedule_var,
                *(m.interval_var(start, end) for start, end in forbidden)]))

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.add(cp.sum(schedule_presence_flat) <= args.nexp)

    log.info('adding constraint: a pixel is observed if it is in any field')
    indices = [[] for _ in range(healpix.npix)]
    with tqdm(total=len(centers) * len(rolls)) as progress:
        for i, grid_i in enumerate(
                mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    indices[k].append((i, j))
                progress.update()
    for lhs_indices, rhs in zip(tqdm(indices), pixel_observed):
        m.add(cp.sum(schedule_presence[i][j] for i, j in lhs_indices)
              >= rhs)

    log.info('adding objective')
    m.maximize(m.scal_prod(pixel_observed, prob))

    log.info('solving')
    kwargs = {'LogVerbosity': 'Verbose'}
    if args.timeout is not None:
        kwargs['TimeLimit'] = args.timeout
    if args.jobs is None:
        kwargs['Workers'] = 'Auto'
    else:
        kwargs['Workers'] = args.jobs
    stopwatch = Stopwatch()
    stopwatch.start()
    solution = m.solve(**kwargs)
    stopwatch.stop()

    log.info('extracting results')
    objective_value, = solution.get_objective_values()
    result_centers, result_rolls, result_times = zip(
        *((center, roll, times[solution.get_value(schedule__)[0]])
            for schedule_, center in zip(schedule, centers)
            for schedule__, roll in zip(schedule_, rolls)
            if solution.get_value(schedule__)))
    result = Table(
        data={
            'time': result_times,
            'exptime': np.repeat(args.exptime, len(result_times)),
            'location': mission.orbit(Time(result_times)),
            'center': result_centers,
            'roll': result_rolls
        },
        descriptions={
            'time': 'Start time of observation',
            'exptime': 'Exposure time',
            'location': 'Location of the spacecraft',
            'center': "Pointing of the center of the spacecraft's FOV",
            'roll': 'Roll angle of spacecraft, position angle of FOV',
        },
        meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': objective_value,
            'status': solution.get_solve_status(),
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
