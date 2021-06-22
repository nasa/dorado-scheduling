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
        '--scheduler', choices=('discrete-time', 'continuous-time-slew'),
        default='discrete-time', help='Scheduling algorithm')
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

    from astropy_healpix import HEALPix
    from astropy.coordinates import ICRS
    from astropy.io import fits
    from astropy.time import Time
    from astropy.table import Table
    from docplex.mp.context import Context
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np

    from ..utils import shlex_join

    if args.scheduler == 'discrete-time':
        from ..schedulers.discrete_time import schedule
    elif args.scheduler == 'continuous-time-slew':
        from ..schedulers.continuous_time_slew import schedule
    else:
        raise AssertionError('this code should not be reached')

    mission = getattr(_mission, args.mission)
    healpix = HEALPix(args.nside, order='nested', frame=ICRS())

    # Read multi-order sky map and rasterize to working resolution
    event_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, healpix.level)['PROB']

    # Set up pointing grid
    if args.skygrid_file is not None:
        centers = Table.read(args.skygrid_file, format='ascii.ecsv')['center']
    else:
        centers = getattr(skygrid, args.skygrid_method.replace('-', '_'))(
            args.skygrid_step)
    rolls = np.arange(0, 360, args.roll_step.to_value(u.deg)) * u.deg

    # Set up time grid
    with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
        nexp = int(((mission.min_overhead + args.duration) /
                    (mission.min_overhead + args.exptime)).to(
                        u.dimensionless_unscaled))
        if args.nexp is not None and args.nexp < nexp:
            nexp = args.nexp
        exptime = args.exptime.to(u.s)
        times = event_time + args.delay + np.arange(
            0, args.duration.to_value(u.s), args.time_step.to_value(u.s)) * u.s

    # Configure solver context
    context = Context.make_default_context()
    context.solver.log_output = True
    # Disable the solution pool. We are not examining multiple solutions,
    # and the solution pool can grow to take up a lot of memory.
    context.cplex_parameters.mip.pool.capacity = 0
    if args.timeout is not None:
        context.cplex_parameters.timelimit = args.timeout
        # Since we have a time limit,
        # emphasize finding good feasible solutions over proving optimality.
        context.cplex_parameters.emphasis.mip = 1
    if args.jobs is None:
        context.cplex_parameters.threads = 0
    else:
        context.cplex_parameters.threads = args.jobs

    stopwatch = Stopwatch()
    stopwatch.start()
    result = schedule(
        mission, prob, healpix, centers, rolls, times, exptime, nexp, context)
    stopwatch.stop()

    result.meta['cmdline'] = shlex_join(sys.argv)
    result.meta['real'] = stopwatch.real
    result.meta['user'] = stopwatch.user
    result.meta['sys'] = stopwatch.sys
    result.write(args.output, format='ascii.ecsv')

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
