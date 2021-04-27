#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Command line interface."""
import logging

from astropy import units as u
from ligo.skymap.tool import ArgumentParser, FileType

from .. import mission as _mission
from .. import skygrid
from ..units import equivalencies

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()

    group = p.add_argument_group(
        'Problem setup options',
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
        'Discretization options',
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
    group.add_argument(
        '--number_of_orbits', type=int, default=1,
        help='Number of orbits to simulate')

    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--skygrid-method', default='healpix',
        choices=[key.replace('_', '-') for key in skygrid.__all__],
        help='Sky grid method')
    group.add_argument(
        '--tiles', metavar='TILES.ecsv',
        type=FileType('rb'),
        help='tiles filename')

    p.add_argument(
        '--nside', type=int, default=32, help='HEALPix sampling resolution')
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
    # import shlex
    import sys

    from astropy_healpix import HEALPix
    from astropy.coordinates import ICRS
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table, QTable
    from docplex.mp.model import Model
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from scipy.signal import convolve
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
            0, args.number_of_orbits * args.duration.to_value(u.s),
            args.time_step.to_value(u.s)) * u.s
    rolls = np.arange(0, 90, args.roll_step.to_value(u.deg)) * u.deg

    if args.tiles is not None:
        centers = QTable.read(args.tiles, format='ascii.ecsv')['center']
    else:
        centers = getattr(skygrid, args.skygrid_method.replace('-', '_'))(
            args.skygrid_step)

    log.info('evaluating field of regard')
    not_regard = convolve(
        ~mission.get_field_of_regard(centers, times, jobs=args.jobs),
        np.ones(time_steps_per_exposure)[:, np.newaxis],
        mode='valid', method='direct')

    log.info('generating model')
    m = Model()
    if args.jobs is not None:
        m.context.cplex_parameters.threads = args.jobs

    log.info('adding variable: observing schedule')
    shape = (len(centers), len(rolls), not_regard.shape[0])
    schedule = np.reshape(m.binary_var_list(np.prod(shape)), shape)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = np.asarray(m.binary_var_list(healpix.npix))

    log.info('adding variable: whether a given field is used')
    field_used = np.reshape(m.binary_var_list(np.prod(shape[:2])), shape[:2])

    log.info('adding variable: whether a given time step is used')
    time_used = np.asarray(m.binary_var_list(shape[2]))

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.add_constraint_(m.sum(time_used) <= args.nexp)

    log.info('adding constraint: only observe one field at a time')
    m.add_constraints_(
        m.sum(schedule[..., i].ravel()) <= 1 for i in tqdm(range(shape[2]))
    )
    m.add_equivalences(
        time_used,
        [m.sum(schedule[..., i].ravel()) >= 1 for i in tqdm(range(shape[2]))]
    )
    m.add_constraints_(
        m.sum(time_used[i:i+time_steps_per_exposure]) <= 1
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
    indices = [[] for _ in range(healpix.npix)]
    with tqdm(total=len(centers) * len(rolls)) as progress:
        for i, grid_i in enumerate(
                mission.fov.footprint_healpix_grid(healpix, centers, rolls)):
            for j, grid_ij in enumerate(grid_i):
                for k in grid_ij:
                    indices[k].append((i, j))
                progress.update()
    m.add_constraints_(
        m.sum(field_used[lhs_index] for lhs_index in lhs_indices) >= rhs
        for lhs_indices, rhs in zip(tqdm(indices), pixel_observed)
    )

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(not_regard)
    m.add_constraint_(m.sum(schedule[j, :, i].ravel()) <= 0)

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

    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        data={
            'time': times[itime],
            'exptime': np.repeat(args.exptime, len(times[itime])),
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
    result.write(args.output, format='ascii.ecsv')

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
