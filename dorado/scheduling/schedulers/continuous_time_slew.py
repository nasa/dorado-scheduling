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

from astropy.coordinates import EarthLocation
from astropy.table import Table
from astropy import units as u
import numpy as np
from synphot import BlackBody1D, SourceSpectrum
from tqdm.auto import tqdm

from . import Model
from ..utils import nonzero_intervals

log = getLogger(__name__)


def schedule(mission, prob, healpix, centers, rolls, times, exptime, nexp,
             apparent_magnitude, sensitivity, context=None):
    jobs = 1 if context is None else (
        context.cplex_parameters.threads.get() or None)
    if len(rolls) > 1:
        raise NotImplementedError('not handling roll angles yet')

    log.info('evaluating field of regard')
    regard = mission.get_field_of_regard(centers, times, jobs=jobs)

    # Discard fields that are never in the field of regard.
    keep = regard.any(axis=0)
    centers = centers[keep]
    regard = regard[:, keep]

    log.info('calculating field footprints')
    field_indices_by_pix = [[] for _ in range(healpix.npix)]
    field_prob = np.empty((len(centers), len(rolls)))
    for i, grid_i in enumerate(
            mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
        for j, grid_ij in enumerate(grid_i):
            field_prob[i, j] = prob[grid_ij].sum()
            for k in grid_ij:
                field_indices_by_pix[k].append((i, j))

    # Keep only the top n fields.
    n = 30
    # FIXME: not handling rolls here
    i = np.argpartition(-field_prob[:, 0], n)[:n]
    centers = centers[i]
    regard = regard[:, i]

    log.info('calculating field footprints')
    field_indices_by_pix = [[] for _ in range(healpix.npix)]
    field_prob = np.empty((len(centers), len(rolls)))
    for i, grid_i in enumerate(
            mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
        for j, grid_ij in enumerate(grid_i):
            field_prob[i, j] = prob[grid_ij].sum()
            for k in grid_ij:
                field_indices_by_pix[k].append((i, j))

    log.info('evaluating exposure time map')
    source_spectrum = SourceSpectrum(
        BlackBody1D, temperature=10000 * u.K
    ).normalize(
        apparent_magnitude, sensitivity.band
    )
    exptime_map = sensitivity.get_exptime(
        source_spectrum,
        snr=5,
        coord=healpix.healpix_to_skycoord(np.arange(healpix.npix)),
        time=times[0],
        night=False,
        redden=True).to_value(u.s)

    min_exptime_s = 120.0
    duration_s = (times[-1] - times[0]).to_value(u.s)

    log.info('generating model')
    m = Model(context=context)

    dt = (times - times[0]).to_value(u.s)
    lb = dt[regard.argmax(axis=0)]
    ub = dt[regard.shape[0] - 1 - regard[::-1].argmax(axis=0)]

    log.info('variable: field start times')
    field_start_time = m.continuous_var_array(
        (len(centers), len(rolls)), lb=lb, ub=ub)

    log.info('variable: field end times')
    field_end_time = m.continuous_var_array(
        (len(centers), len(rolls)), lb=lb, ub=ub)

    log.info('variable: field exposure times')
    field_exp_time = m.semicontinuous_var_array(
        (len(centers), len(rolls)), lb=min_exptime_s, ub=duration_s)

    log.info('variable: whether a given pixel is used')
    pix_used = m.binary_var_array(healpix.npix)

    m.add_(
        field_exp_time[i, j] <= field_end_time[i, j] - field_start_time[i, j]
        for i in range(len(centers)) for j in range(len(rolls)))

    log.info('constraint: a pixel is used if it is in any used fields')
    # Make the Venn diagram of the footprints of all of the fields.
    key = itemgetter(1)
    region_ipix, region_max_exp_time = zip(*(
        (
            np.asarray(next(zip(*group))),
            m.max(field_exp_time[field_index] for field_index in field_indices)
        ) for field_indices, group
        in groupby(sorted(enumerate(field_indices_by_pix), key=key), key)
        if len(field_indices) > 0))
    # Finally, create variables and constraints.
    region_exp_time = m.semicontinuous_var_array(
        len(region_ipix), lb=min_exptime_s, ub=duration_s)
    for pix, field_indices in zip(pix_used, field_indices_by_pix):
        if len(field_indices) == 0:
            pix.set_ub(0)
    m.add_(
        region_exp_time[i] <= region_max_exp_time[i]
        for i in range(len(region_ipix)))
    m.add_indicator_constraints_(
        pix_used[ipix] >> (region_exp_time[i] >= exptime_map[ipix])
        for i, pixels in enumerate(region_ipix)
        for ipix in pixels)

    seq = m.binary_var_array(
        (len(centers), len(rolls), len(centers), len(rolls)))

    log.info('constraints: sequencing')
    # FIXME: indexing does not handle roll angles yet
    m.add_indicator_constraints_(
        seq[i1, j1, i2, j2]
        >> (field_end_time[i1, j1] <= field_start_time[i2, j2])
        for i1 in tqdm(range(len(centers))) for j1 in range(len(rolls))
        for i2 in range(len(centers)) for j2 in range(len(rolls))
        if i1 < i2)
    m.add_indicator_constraints_(
        (seq[i1, j1, i2, j2] == 0)
        >> (field_end_time[i2, j2] <= field_start_time[i1, j1])
        for i1 in tqdm(range(len(centers))) for j1 in range(len(rolls))
        for i2 in range(len(centers)) for j2 in range(len(rolls))
        if i1 < i2)

    # log.info('evaluating overhead')
    # overhead_s = mission.overhead(
    #     centers[:, np.newaxis, np.newaxis, np.newaxis],
    #     centers[np.newaxis, np.newaxis, :, np.newaxis],
    #     rolls[np.newaxis, :, np.newaxis, np.newaxis],
    #     rolls[np.newaxis, np.newaxis, np.newaxis, :]).to_value(u.s)

    m.maximize(m.scal_prod(pix_used, prob))

    log.info('constraint: field of regard')
    for j in range(len(centers)):
        for k in range(len(rolls)):
            # FIXME: not roll dependent yet
            intervals = (times - times[0]).to_value(u.s)[
                nonzero_intervals(regard[:, j])]

            if len(intervals) == 0:
                raise ValueError('Should not be reached')
            elif len(intervals) > 1:
                # The field is within the field of regard during two or more
                # disjoint intervals, so introduce additional decision
                # variables to decide which interval.
                interval_start, interval_end = intervals.T
                interval_choice = m.binary_var_array(len(intervals))
                m.add_(m.logical_or(*interval_choice) == 1)
                m.add_indicator_constraints_(
                    interval_choice[i] >>
                    (field_start_time[j, k] >= interval_start[i])
                    for i in range(len(intervals)))
                m.add_indicator_constraints_(
                    interval_choice[i] >>
                    (field_end_time[j, k] <= interval_end[i])
                    for i in range(len(intervals)))

    log.info('solving')
    solution = m.solve(log_output=True)

    log.info('extracting results')
    if solution is None:
        field_start_time_value = np.zeros(field_start_time.shape)
        field_exp_time_value = np.zeros(field_exp_time.shape)
        objective_value = 0.0
    else:
        field_start_time_value = np.reshape(
            solution.get_values(field_start_time.ravel()),
            field_start_time.shape)
        field_exp_time_value = np.reshape(
            solution.get_values(field_exp_time.ravel()),
            field_exp_time.shape)
        objective_value = m.objective_value

    i, j = np.nonzero(field_exp_time_value)
    field_start_time_value = times[0] + field_start_time_value * u.s
    result = Table(
        data={
            'time': field_start_time_value[i, j],
            'exptime': field_exp_time_value[i, j],
            'location': (
                # FIXME: remove this workaround for empty EarthLocation once
                # https://github.com/astropy/astropy/issues/11454 is released
                mission.orbit(field_start_time_value).earth_location[i, j]
                if len(i) > 0 else EarthLocation([], [])),
            'center': centers[i],
            'roll': rolls[j]
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
