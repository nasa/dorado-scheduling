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


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    import sys
    import warnings

    from astropy_healpix import HEALPix
    from astropy.coordinates import ICRS
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    from cplex.callbacks import LazyConstraintCallback
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from tqdm import tqdm

    from ..schedulers import Model
    from ..utils import nonzero_intervals, shlex_join

    mission = getattr(_mission, args.mission)
    healpix = HEALPix(args.nside, order='nested', frame=ICRS())

    log.info('reading sky map')
    # Read multi-order sky map and rasterize to working resolution
    event_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, healpix.level)['PROB']

    # Set up grids
    with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
        nexp = int(((mission.min_overhead + args.duration) /
                    (mission.min_overhead + args.exptime)).to(
                        u.dimensionless_unscaled))
        if args.nexp is not None and args.nexp < nexp:
            nexp = args.nexp
        min_delay_s = (mission.min_overhead + args.exptime).to_value(u.s)
        exptime_s = args.exptime.to_value(u.s)
        duration_s = args.duration.to_value(u.s)
        time_step_s = args.time_step.to_value(u.s)
        start_time = event_time + args.delay
        times = start_time + np.arange(
            0, duration_s, time_step_s) * u.s
    rolls = np.arange(0, 90, args.roll_step.to_value(u.deg)) * u.deg

    if len(rolls) > 1:
        warnings.warn(
            'Slew constraints for varying roll angles are not yet implemented')

    if args.skygrid_file is not None:
        centers = Table.read(args.skygrid_file, format='ascii.ecsv')['center']
    else:
        centers = getattr(skygrid, args.skygrid_method.replace('-', '_'))(
            args.skygrid_step)

    log.info('evaluating field of regard')
    regard = mission.get_field_of_regard(centers, times, jobs=args.jobs)

    log.info('generating model')
    m = Model()
    if args.timeout is not None:
        m.set_time_limit(args.timeout)
    if args.jobs is not None:
        m.context.cplex_parameters.threads = args.jobs

    log.info('variable: start time for each observation')
    obs_start_time = m.continuous_var_array(
        nexp, lb=0, ub=duration_s - exptime_s)

    log.info('variable: field selection for each observation')
    obs_field = m.binary_var_array((nexp, len(centers), len(rolls)))

    log.info('variable: whether a given observation is used')
    obs_used = m.binary_var_array(nexp)

    log.info('variable: whether a given field is used')
    field_used = m.binary_var_array((len(centers), len(rolls)))

    log.info('variable: whether a given pixel is observed')
    pix_used = m.binary_var_array(healpix.npix)

    log.info('constraint: at most one field is used for each observation')
    m.add_(m.sum(obs_field[i].ravel()) <= obs_used[i] for i in range(nexp))

    log.info('constraint: consecutive observations are used')
    m.add_(obs_used[i] >= obs_used[i + 1] for i in range(nexp - 1))

    log.info('constraint: a field is used if it is chosen for an observation')
    m.add_(
        m.sum(obs_field[:, j, k]) >= field_used[j, k]
        for j in tqdm(range(len(centers))) for k in range(len(rolls)))

    log.info('constraint: a pixel is used if it is in any used fields')
    indices = [[] for _ in range(healpix.npix)]
    with tqdm(total=len(centers) * len(rolls)) as progress:
        for i, grid_i in enumerate(
                mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    indices[k].append((i, j))
                progress.update()
    m.add_(
        m.sum(field_used[lhs_index] for lhs_index in lhs_indices) >= rhs
        for lhs_indices, rhs in zip(tqdm(indices), pix_used))

    log.info('constraint: observations do not overlap')
    m.add_indicator_constraints_(
        obs_used[i + 1] >>
        (obs_start_time[i + 1] - obs_start_time[i] >= min_delay_s)
        for i in range(nexp - 1))

    log.info('constraint: field of regard')
    for j in tqdm(range(len(centers))):
        for k in range(len(rolls)):
            # FIXME: not roll dependent yet
            intervals = nonzero_intervals(regard[:, j]) * time_step_s

            if len(intervals) == 0:
                # The field is always outside the field of regard,
                # so disallow it entirely.
                m.add_(m.sum(obs_field[:, j, k]) <= 0)
            elif len(intervals) == 1:
                # The field is within the field of regard during a single
                # contigous interval, so require the observaiton to be within
                # that interval.
                (interval_start, interval_end), = intervals
                m.add_indicator_constraints_(
                    obs_field[i, j, k] >>
                    (obs_start_time[i] >= interval_start)
                    for i in range(nexp))
                m.add_indicator_constraints_(
                    obs_field[i, j, k] >>
                    (obs_start_time[i] <= interval_end - exptime_s)
                    for i in range(nexp))
            else:
                # The field is within the field of regard during two or more
                # disjoint intervals, so introduce additional decision
                # variables to decide which interval.
                interval_start, interval_end = intervals.T
                interval_choice = m.binary_var_array((nexp, len(intervals)))
                m.add_indicator_constraints_(
                    obs_field[i, j, k] >> (m.sum(interval_choice[i]) >= 1)
                    for i in range(nexp))
                m.add_indicator_constraints_(
                    interval_choice[i, i1] >>
                    (obs_start_time[i] >= interval_start[i1])
                    for i in range(nexp) for i1 in range(len(intervals)))
                m.add_indicator_constraints_(
                    interval_choice[i, i1] >>
                    (obs_start_time[i] <= interval_end[i1] - exptime_s)
                    for i in range(nexp) for i1 in range(len(intervals)))

    class SlewConstraintCallback(ConstraintCallbackMixin,
                                 LazyConstraintCallback):

        def __init__(self, env):
            LazyConstraintCallback.__init__(self, env)
            ConstraintCallbackMixin.__init__(self)

        def __call__(self):
            # Reconstruct partial solution
            sol = self.make_solution_from_watched()

            # Determine which fields are selected
            i, j, k = np.nonzero(np.reshape(sol.get_values(obs_field.ravel()),
                                            obs_field.shape))

            # Calculate overhead + exptime between each pair of fields
            coords = centers[j]
            dt_array = mission.overhead(coords[1:], coords[:-1]).to_value(u.s)
            dt_array += exptime_s

            lhs_array = obs_start_time[i]
            rhs_array = obs_field[i, j, k]
            # This is a big-M formulation of the indicator constraint:
            # (rhs1 & rhs0) >> lhs1 - lhs0 >= dt
            cons = [
                lhs1 - lhs0 - dt >= duration_s * (rhs1 + rhs0 - 2)
                for lhs0, lhs1, rhs0, rhs1, dt
                in zip(lhs_array[:-1], lhs_array[1:],
                       rhs_array[:-1], rhs_array[1:], dt_array)]

            for _, lhs, sense, rhs in self.get_cpx_unsatisfied_cts(cons, sol):
                self.add(lhs, sense, rhs)

    cb = m.register_callback(SlewConstraintCallback)
    cb.register_watched_vars(obs_field.ravel())
    cb.register_watched_vars(obs_start_time)

    log.info('adding objective')
    m.maximize(m.scal_prod(pix_used, prob))

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    solution = m.solve(log_output=True)
    stopwatch.stop()

    log.info('extracting results')
    if solution is None:
        obs_field_value = np.zeros(obs_field.shape, dtype=bool)
        obs_start_time_value = np.zeros(obs_start_time.shape)
        objective_value = 0.0
    else:
        obs_field_value = np.asarray(
            solution.get_values(obs_field.ravel()), dtype=bool
        ).reshape(
            obs_field.shape
        )
        obs_start_time_value = np.asarray(solution.get_values(obs_start_time))
        objective_value = m.objective_value
    obs_start_time_value = start_time + obs_start_time_value * u.s

    i, j, k = np.nonzero(obs_field_value)
    result = Table(
        data={
            'time': obs_start_time_value[i],
            'exptime': np.repeat(args.exptime, len(obs_start_time_value[i])),
            'location': mission.orbit(obs_start_time_value).earth_location[i],
            'center': centers[j],
            'roll': rolls[k]
        },
        descriptions={
            'time': 'Start time of observation',
            'exptime': 'Exposure time',
            'location': 'Location of the spacecraft',
            'center': "Pointing of the center of the spacecraft's FOV",
            'roll': 'Roll angle of spacecraft, position angle of FOV',
        },
        meta={
            'cmdline': shlex_join(sys.argv),
            'prob': objective_value,
            'status': m.solve_status.name,
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
