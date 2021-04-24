#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Command line interface."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

from .. import mission as _mission

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('config', help='config file')
    p.add_argument('mission', choices=set(_mission.__all__) - {'Mission'},
                   default='dorado', help='Mission configuration')
    p.add_argument('-n', '--nexp', type=int, help='Number of exposures')
    p.add_argument('-s', '--start_time', type=str,
                   default='2020-01-01T00:00:00')
    p.add_argument('--output', '-o', metavar='OUTPUT.ecsv',
                   type=FileType('w'), default='-',
                   help='output filename')
    p.add_argument('-j', '--jobs', type=int, default=1, const=None, nargs='?',
                   help='Number of threads')

    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    # import shlex
    import sys

    from astropy_healpix import nside_to_level
    from astropy.time import Time
    from astropy.table import Table, QTable
    from astropy import units as u
    import configparser
    from docplex.mp.model import Model
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from scipy.signal import convolve
    from tqdm import tqdm

    from ..models import SurveyModel

    config = configparser.ConfigParser()
    config.read(args.config)

    mission = getattr(_mission, args.mission)
    tiles = QTable.read(config["survey"]["tilesfile"], format='ascii.ecsv')

    exposure_time = float(config["survey"]["exposure_time"]) * u.minute
    steps_per_exposure =\
        int(config["survey"]["time_steps_per_exposure"])
    number_of_orbits = int(config["survey"]["number_of_orbits"])

    survey_model = SurveyModel(mission=mission,
                               exposure_time=exposure_time,
                               time_steps_per_exposure=steps_per_exposure,
                               number_of_orbits=number_of_orbits,
                               centers=tiles["center"])

    log.info('reading sky map')
    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(args.start_time, format='isot')
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap,
                     nside_to_level(survey_model.healpix.nside))['PROB']
    if survey_model.healpix.order == 'ring':
        prob = prob[survey_model.healpix.ring_to_nested(np.arange(len(prob)))]

    times = np.arange(survey_model.time_steps) *\
        survey_model.time_step_duration + start_time

    log.info('generating model')
    m = Model()
    m.set_time_limit(300)
    if args.jobs is not None:
        m.context.cplex_parameters.threads = args.jobs

    log.info('adding variable: observing schedule')
    shape = (len(survey_model.centers),
             survey_model.time_steps -
             survey_model.time_steps_per_exposure + 1)
    schedule = np.reshape(m.binary_var_list(np.prod(shape)), shape)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = np.asarray(m.binary_var_list(survey_model.healpix.npix))

    log.info('adding variable: whether a given field is used')
    field_used = np.asarray(m.binary_var_list(shape[0]))

    log.info('adding variable: whether a given time step is used')
    time_used = np.asarray(m.binary_var_list(shape[1]))

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.add_constraint_(m.sum(time_used) <= args.nexp)

    log.info('adding constraint: only observe one field at a time')
    m.add_constraints_(
        m.sum(schedule[..., i].ravel()) <= 1 for i in tqdm(range(shape[1]))
    )
    m.add_equivalences(
        time_used,
        [m.sum(schedule[..., i].ravel()) >= 1 for i in tqdm(range(shape[1]))]
    )
    m.add_constraints_(
        m.sum(time_used[i:i+survey_model.time_steps_per_exposure]) <= 1
        for i in tqdm(range(schedule.shape[-1]))
    )

    log.info('adding constraint: a pixel is observed if it is in any field')
    m.add_constraints_(
        m.sum(lhs) >= rhs
        for lhs, rhs in zip(
            tqdm(schedule.reshape(field_used.size, -1)),
            field_used.ravel()
        )
    )

    indices = [[] for _ in range(survey_model.healpix.npix)]
    with tqdm(total=len(survey_model.centers)) as progress:
        for i, center in enumerate(survey_model.centers):
            grid_ij = survey_model.mission.fov.footprint_healpix(
                survey_model.healpix, center)
            for k in grid_ij:
                indices[k].append(i)
            progress.update()
    m.add_constraints_(
        m.sum(field_used[lhs_index] for lhs_index in lhs_indices) >= rhs
        for lhs_indices, rhs in zip(tqdm(indices), pixel_observed)
    )

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(
            ~survey_model.get_field_of_regard(times, jobs=args.jobs),
            np.ones(survey_model.time_steps_per_exposure)[:, np.newaxis],
            mode='valid', method='direct'))
    m.add_constraint_(m.sum(schedule[j, i].ravel()) <= 0)

    log.info('adding objective')
    m.maximize(m.scal_prod(pixel_observed, prob))

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    solution = m.solve(log_output=True)
    stopwatch.stop()

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

    ipix, itime = np.nonzero(schedule_flags)
    result = Table(
        {
            'time': times[itime],
            'center': survey_model.centers[ipix],
        }, meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': objective_value,
            'status': m.solve_status.name,
            'real': stopwatch.real,
            'user': stopwatch.user,
            'sys': stopwatch.sys
        }
    )
    result.sort('time')
    result.write(args.output.name, format='ascii.ecsv')

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
