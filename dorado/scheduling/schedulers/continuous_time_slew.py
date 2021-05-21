#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from itertools import groupby
from logging import getLogger
from operator import itemgetter
import warnings

from astropy.table import Table
from astropy import units as u
import numpy as np

from . import Model
from ..utils import nonzero_intervals

log = getLogger(__name__)


def schedule(mission, prob, healpix, centers, rolls, times, exptime, nexp,
             context=None):
    if len(rolls) > 1:
        warnings.warn(
            'Slew constraints for varying roll angles are not yet implemented')

    min_delay_s = (mission.min_overhead + exptime).to_value(u.s)
    exptime_s = exptime.to_value(u.s)
    duration_s = (times[-1] - times[0]).to_value(u.s)
    jobs = 1 if context is None else (
        context.cplex_parameters.threads.get() or None)

    log.info('evaluating field of regard')
    regard = mission.get_field_of_regard(centers, times, jobs=jobs)

    log.info('generating model')
    m = Model(context)

    log.info('variable: start time for each observation')
    obs_start_time = m.continuous_var_array(
        nexp, lb=0, ub=duration_s - exptime_s)

    log.info('variable: field selection for each observation')
    obs_field = m.binary_var_array((nexp, len(centers), len(rolls)))

    log.info('variable: whether a given observation is used')
    obs_used = m.binary_var_array(nexp)

    log.info('variable: whether a given field is used')
    field_used = m.binary_var_array((len(centers), len(rolls)))

    log.info('constraint: at most one field is used for each observation')
    m.add_(m.sum(obs_field[i].ravel()) <= obs_used[i] for i in range(nexp))

    log.info('constraint: consecutive observations are used')
    m.add_(obs_used[i] >= obs_used[i + 1] for i in range(nexp - 1))

    log.info('constraint: a field is used if it is chosen for an observation')
    m.add_(
        m.sum(obs_field[:, j, k]) >= field_used[j, k]
        for j in range(len(centers)) for k in range(len(rolls)))

    log.info('constraint: a pixel is used if it is in any used fields')
    # First, make a table of the fields that contain each pixel.
    field_indices_by_pix = [[] for _ in range(healpix.npix)]
    for i, grid_i in enumerate(
            mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
        for j, grid_ij in enumerate(grid_i):
            for k in grid_ij:
                field_indices_by_pix[k].append((i, j))
    # Next, make the Venn diagram of the footprints of all of the fields.
    key = itemgetter(1)
    coefficients, lhss = zip(*(
        (
            prob[np.asarray(next(zip(*group)))].sum(),
            m.sum(field_used[field_index] for field_index in field_indices)
        ) for field_indices, group
        in groupby(sorted(enumerate(field_indices_by_pix), key=key), key)))
    # Finally, create variables and constraints.
    pix_used = m.binary_var_array(len(coefficients))
    m.add_(lhs >= rhs for lhs, rhs in zip(lhss, pix_used))
    m.maximize(m.scal_prod(pix_used, coefficients))

    log.info('constraint: observations do not overlap')
    m.add_indicator_constraints_(
        obs_used[i + 1] >>
        (obs_start_time[i + 1] - obs_start_time[i] >= min_delay_s)
        for i in range(nexp - 1))

    log.info('constraint: field of regard')
    for j in range(len(centers)):
        for k in range(len(rolls)):
            # FIXME: not roll dependent yet
            intervals = (times - times[0]).to_value(u.s)[
                nonzero_intervals(regard[:, j])]

            if len(intervals) == 0:
                # The field is always outside the field of regard,
                # so disallow it entirely.
                m.add_(m.sum(obs_field[:, j, k]) <= 0)
            elif len(intervals) == 1:
                # The field is within the field of regard during a single
                # contiguous interval, so require the observation to be within
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

    def callback(sol):
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
        return [
            lhs1 - lhs0 - dt >= duration_s * (rhs1 + rhs0 - 2)
            for lhs0, lhs1, rhs0, rhs1, dt
            in zip(lhs_array[:-1], lhs_array[1:],
                   rhs_array[:-1], rhs_array[1:], dt_array)]

    m.set_lazy_constraint_callback(callback, obs_field.ravel(), obs_start_time)

    log.info('solving')
    solution = m.solve(log_output=True)

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
    obs_start_time_value = times[0] + obs_start_time_value * u.s

    i, j, k = np.nonzero(obs_field_value)
    result = Table(
        data={
            'time': obs_start_time_value[i],
            'exptime': np.repeat(exptime, len(obs_start_time_value[i])),
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
            'prob': objective_value,
            'status': m.solve_details.status,
            'solve_time': m.solve_details.time
        }
    )
    result.sort('time')
    return result
