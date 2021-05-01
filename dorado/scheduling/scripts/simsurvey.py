#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Simulate full survey."""

import glob
import os
import logging
import numpy as np

from ligo.skymap.tool import ArgumentParser, FileType

from astropy.time import Time
from astropy.table import QTable, vstack
from astropy_healpix import nside_to_level
from astropy.coordinates import SkyCoord
from astropy import units as u

from ligo.skymap.io import read_sky_map, write_sky_map
from ligo.skymap.bayestar import rasterize

from .. import mission as _mission
from .. import skygrid
from ..units import equivalencies

np.random.seed(0)

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling-simsurvey')
    p.add_argument('config', help='config file')
    group = p.add_argument_group(
        'problem setup options',
        'Options that control the problem setup')
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
        '--time-step', type=u.Quantity, default='10 min',
        help='Model time step (any time units)')
    group.add_argument(
        '--roll-step', type=u.Quantity, default='90 deg',
        help='Roll angle step (any angle units)')
    group.add_argument(
        '--skygrid-step', type=u.Quantity, default='0.0011 sr',
        help='Sky grid resolution (any solid angle units')

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--skygrid-method', default='healpix',
        choices=[key.replace('_', '-') for key in skygrid.__all__],
        help='Sky grid method')
    group.add_argument(
        '--skygrid-file', metavar='TILES.ecsv',
        type=FileType('rb'),
        help='tiles filename')

    p.add_argument('-s', '--start_time', type=str,
                   default='2020-01-01T00:00:00')
    p.add_argument(
        '--duration-survey', type=u.Quantity, default='10 orbit',
        help='Duration of survey observing plan (any time units)')
    p.add_argument('--output', '-o',
                   type=str, default='simsurvey',
                   help='output survey')
    p.add_argument(
        '--nside', type=int, default=32, help='HEALPix sampling resolution')
    p.add_argument('--timeout', type=int,
                   default=300, help='Impose timeout on solutions')

    p.add_argument('--gw-folder', '-g',
                   type=str, default='examples',
                   help='folder with GW fits files')

    p.add_argument("--doDust", help="load CSV", action="store_true")
    p.add_argument("--doAnimate", help="load CSV", action="store_true")
    p.add_argument("--doMetrics", help="load CSV", action="store_true")
    p.add_argument("--doPlotSkymaps", help="load CSV", action="store_true")
    p.add_argument("--doSlicer", help="load CSV", action="store_true")
    p.add_argument("--doOverlap", help="load CSV", action="store_true")

    return p


def get_observed(latest_time, mission, healpix, schedulenames, prob):

    ras, decs, tts = [], [], []
    for schedulename in schedulenames:
        schedule = QTable.read(schedulename, format='ascii.ecsv')
        for row in schedule:
            ras.append(row["center"].ra.deg)
            decs.append(row["center"].dec.deg)
            tts.append(row["time"])

    probscale = np.ones(prob.shape)
    for ra, dec, tt in zip(ras, decs, tts):
        ipix = mission.fov.footprint_healpix(healpix, SkyCoord(ra*u.deg,
                                                               dec*u.deg))
        dt = latest_time - tt
        tau = 60.0
        scale = 1 - np.exp(-dt.jd/tau)
        probscale[ipix] = probscale[ipix] * scale

    prob = prob*probscale
    prob = prob / np.sum(prob)

    return prob


def compute_overlap(mission, centers, healpix):
    res = healpix.pixel_area.to_value(u.arcmin**2)
    ipix = {}
    for ii, cent1 in enumerate(centers):
        ipix[ii] = mission.fov.footprint_healpix(healpix, cent1)
    overlaps = []
    for ii, cent1 in enumerate(centers):
        if np.mod(ii, 100) == 0:
            print('Computing %d/%d tiles' % (ii, len(centers)))
        overlap = 0.0
        for jj, cent2 in enumerate(centers):
            if ii <= jj:
                continue
            over = np.intersect1d(ipix[ii], ipix[jj])
            overlap = np.max([overlap, len(over)*res])
        overlaps.append(overlap)
    print('max overlap: %.5f arcmin^2' % (np.max(overlaps)))


def merge_tables(schedulenames):

    for ii, schedulename in enumerate(schedulenames):
        schedule = QTable.read(schedulename, format='ascii.ecsv')
        survey = schedulename.split("/")[-1].split("_")[1]
        fitsfile = schedulename.replace("csv", "fits")
        fitsfile = fitsfile.replace("survey_%s" % survey, "skymap_%s" % survey)
        if len(schedule) == 0:
            continue
        schedule.add_column(survey, name='survey')
        schedule.add_column(fitsfile, name='skymap')
        if ii == 0:
            scheduleall = schedule
        else:
            scheduleall = vstack([scheduleall, schedule])

    scheduleall.sort('time')

    return scheduleall


def main(args=None):
    args = parser().parse_args(args)

    import configparser
    from astropy.coordinates import ICRS
    from astropy_healpix import HEALPix
    from ..fov import FOV

    config = configparser.ConfigParser()
    config.read(args.config)

    mission = getattr(_mission, args.mission)
    healpix = HEALPix(args.nside, order='nested', frame=ICRS())
    tiles = QTable.read(args.skygrid_file, format='ascii.ecsv')
    centers = tiles['center']

    # Set up grids
    with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
        niter = int(np.round(
            (args.duration_survey / args.duration).to_value(
                u.dimensionless_unscaled)))

    coords = healpix.healpix_to_skycoord(np.arange(healpix.npix))

    if args.doOverlap:
        compute_overlap(mission, centers, healpix)

    outdir = args.output
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if args.doDust:
        from dustmaps.planck import PlanckQuery
        from ..dust import Dust

        planck = PlanckQuery()
        dust_properties = Dust(config)

        Ax1 = dust_properties.Ax1
        zeropointDict = dust_properties.zeropointDict
        ebv = planck(coords)
        # Apply dust extinction on the light curve
        A_x = Ax1['NUV'] * ebv

        dustname = '%s/dust.fits' % outdir
        write_sky_map(dustname, A_x, moc=True)

        system_command = 'ligo-skymap-plot %s -o %s --colorbar' % (
            dustname, dustname.replace("fits", "png"))
        os.system(system_command)

        V = 10**(0.6*(zeropointDict['NUV']-A_x))
        V = V / np.max(V)

    quad = QTable.read(config["simsurvey"]["quadfile"], format='ascii.ecsv')
    quad.add_index('field_id')
    quadlen = len(quad)
    quad_fov = float(config["simsurvey"]["quad_field_of_view"])
    quad_fov = FOV.from_rectangle(quad_fov * u.deg)

    start_time = Time(args.start_time, format='isot')

    surveys = config["simsurvey"]["surveys"].split(",")
    weights = [0] + [float(x) for x in
                     config["simsurvey"]["weights"].split(",")]
    weights_cumsum = np.cumsum(weights)

    randvals = np.random.rand(niter)

    gwfits = glob.glob(os.path.join(args.gw_folder, '*.fits')) +\
        glob.glob(os.path.join(args.gw_folder, '*.fits.fz'))

    schedulenames = []
    tind = 0

    for jj in range(niter):

        randval = randvals[jj]
        idx = np.where((weights_cumsum[1:] >= randval) &
                       (weights_cumsum[:-1] <= randval))[0]

        survey = surveys[int(idx)]
        schedulename = '%s/survey_%s_%05d.csv' % (outdir, survey, jj)
        skymapname = '%s/skymap_%s_%05d.fits' % (outdir, survey, jj)
        gifname = '%s/skymap_%s_%05d.gif' % (outdir, survey, jj)

        with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
            times = start_time + args.delay + np.arange(
                0, args.duration.to_value(u.s),
                args.time_step.to_value(u.s)) * u.s

        if os.path.isfile(schedulename):
            schedulenames.append(schedulename)

            orb = mission.orbit
            with u.add_enabled_equivalencies(equivalencies.orbital(orb)):
                times = start_time + args.delay + np.arange(
                    0, args.duration.to_value(u.s),
                    args.time_step.to_value(u.s)) * u.s

            start_time = times[-1] + args.exptime
            if survey == "baseline":
                tind = tind + 1
                tind = np.mod(tind, quadlen)
            continue

        if survey == "galactic_plane":
            prob = (np.abs(coords.galactic.b.deg) <= 15.0)
            prob = prob / prob.sum()

            prob = get_observed(start_time, mission, healpix,
                                schedulenames, prob)
            if args.doDust:
                prob = prob*V

        elif survey == "kilonova":
            n = 0.01 * np.ones(healpix.npix)

            tindex = int(quadlen/2)
            tquad = quad.loc[tindex]
            raquad, decquad = tquad["center"].ra, tquad["center"].dec
            p = quad_fov.footprint_healpix(healpix,
                                           SkyCoord(raquad, decquad))
            n[p] = 1.
            prob = n / np.sum(n)
            if args.doDust:
                prob = prob*V

        elif survey == "GW":
            idx = int(np.floor(len(gwfits)*np.random.rand()))
            skymap = read_sky_map(gwfits[idx],
                                  moc=True)['UNIQ', 'PROBDENSITY']
            prob = rasterize(
                skymap, nside_to_level(healpix.nside))['PROB']
            prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]
            if args.doDust:
                prob = prob*V

        elif survey == "baseline":
            n = 0.01 * np.ones(healpix.npix)

            tquad = quad.loc[tind]
            raquad, decquad = tquad["center"].ra, tquad["center"].dec
            p = quad_fov.footprint_healpix(healpix,
                                           SkyCoord(raquad, decquad))
            n[p] = 1.
            prob = n / np.sum(n)
            prob = get_observed(start_time, mission, healpix,
                                schedulenames, prob)
            if args.doDust:
                prob = prob*V

            tind = tind + 1
            tind = np.mod(tind, quadlen)

        write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

        executable = 'dorado-scheduling'
        system_command = ("%s %s -o %s --mission %s --exptime '%s' "
                          "--time-step '%s' --roll-step '%s' "
                          "--skygrid-file %s --duration '%s' "
                          "--timeout %d") % (
            executable,
            skymapname, schedulename, args.mission,
            str(args.exptime), str(args.time_step), str(args.roll_step),
            args.skygrid_file.name, str(args.duration),
            args.timeout)
        print(system_command)
        os.system(system_command)

        if args.doPlotSkymaps:
            system_command = 'ligo-skymap-plot %s -o %s' % (
                skymapname, skymapname.replace("fits", "png"))
            os.system(system_command)

        schedulenames.append(schedulename)
        start_time = times[-1] + args.exptime

    scheduleall = merge_tables(schedulenames)
    schedulename = '%s/metrics/survey_all.csv' % (outdir)
    skymapname = '%s/metrics/survey_all.fits' % (outdir)
    gifname = '%s/metrics/survey_all.mp4' % (outdir)
    metricsname = '%s/metrics' % (outdir)
    if not os.path.isdir(metricsname):
        os.makedirs(metricsname)

    scheduleall.write(schedulename, format='ascii.ecsv')

    n = np.ones(healpix.npix)
    prob = n / np.sum(n)
    write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

    if args.doMetrics:
        executable = 'dorado-scheduling-survey-metrics'
        system_command = '%s %s %s --mission %s -o %s --skygrid-file %s' % (
            executable, skymapname, schedulename, args.mission,
            metricsname, args.skygrid_file.name)
        print(system_command)
        os.system(system_command)

    if args.doAnimate:
        start_time = scheduleall[0]["time"]
        executable = 'dorado-scheduling-animate-survey'
        system_command = "%s %s %s --mission %s -o %s -s %s --nside %d" % (
            executable, skymapname, schedulename, args.mission,
            gifname, start_time.isot, args.nside)
        print(system_command)
        os.system(system_command)

    if args.doSlicer:
        executable = 'dorado-scheduling-survey-slicer'
        system_command = '%s %s %s %s --mission %s -o %s --nside %d' % (
            executable, args.config, skymapname, schedulename, args.mission,
            metricsname, args.nside)
        print(system_command)
        os.system(system_command)
