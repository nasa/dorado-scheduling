#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
import logging

from astropy.table import Table
from astropy import units as u
import numpy as np
from scipy.signal import convolve
from tqdm import tqdm

from ..schedulers import Model

log = logging.getLogger(__name__)


def schedule(mission, prob, healpix, centers, rolls, times, exptime, nexp,
             context=None):
    jobs = 1 if context is None else (
        context.cplex_parameters.threads.get() or None)

    time_step = times[1] - times[0]
    time_steps_per_exposure = int(np.round(
        (exptime / time_step).to_value(
            u.dimensionless_unscaled)))

    log.info('evaluating field of regard')
    not_regard = convolve(
        ~mission.get_field_of_regard(centers, times, jobs=jobs),
        np.ones(time_steps_per_exposure)[:, np.newaxis],
        mode='valid', method='direct')

    log.info('generating model')
    m = Model(context=context)

    log.info('adding variable: observing schedule')
    schedule = m.binary_var_array(
        (len(centers), len(rolls), not_regard.shape[0]))

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = m.binary_var_array(healpix.npix)

    log.info('adding variable: whether a given field is used')
    field_used = m.binary_var_array(schedule.shape[:2])

    log.info('adding variable: whether a given time step is used')
    time_used = m.binary_var_array(schedule.shape[2])

    if nexp is not None:
        log.info('adding constraint: number of exposures')
        m.add_(m.sum(time_used) <= nexp)

    log.info('adding constraint: only observe one field at a time')
    m.add_(
        m.sum(schedule[..., i].ravel()) <= 1
        for i in tqdm(range(schedule.shape[2])))
    m.add_equivalences(
        time_used,
        [m.sum(schedule[..., i].ravel()) >= 1
         for i in tqdm(range(schedule.shape[2]))]
    )
    m.add_(
        m.sum(time_used[i:i+time_steps_per_exposure]) <= 1
        for i in tqdm(range(schedule.shape[2]))
    )

    log.info('adding constraint: a pixel is observed if it is in any field')
    m.add_(
        m.sum(lhs) >= rhs
        for lhs, rhs in zip(
            tqdm(schedule.reshape(field_used.size, -1)),
            field_used.ravel()
        )
    )
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
        for lhs_indices, rhs in zip(tqdm(indices), pixel_observed)
    )

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(not_regard)
    m.add_(m.sum(schedule[j, :, i].ravel()) <= 0)

    log.info('adding objective')
    m.maximize(m.scal_prod(pixel_observed, prob))

    log.info('solving')
    solution = m.solve()

    log.info('extracting results')
    if solution is None:
        schedule_flags = np.zeros(schedule.shape, dtype=bool)
        objective_value = 0.0
    else:
        schedule_flags = np.asarray(
            solution.get_values(schedule.ravel()), dtype=bool
        ).reshape(
            schedule.shape
        )
        objective_value = m.objective_value

    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        data={
            'time': times[itime],
            'exptime': np.repeat(exptime, len(times[itime])),
            'location': mission.orbit(times).earth_location[itime],
            'center': centers[ipix],
            'roll': rolls[iroll]
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
