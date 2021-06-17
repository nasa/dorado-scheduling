#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Simulate full survey."""

import glob
import copy
import os
import logging
import healpy as hp
import numpy as np
import pandas as pd

from ligo.skymap.tool import ArgumentParser, FileType

from astropy.time import Time
from astropy.table import Table, QTable, vstack
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
        '--roll-step', type=u.Quantity, default='360 deg',
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
    p.add_argument('--gw-too-file',
                   type=str,
                   help='simulations file with GW exposure times')

    p.add_argument(
        '-j', '--jobs', type=int, default=1, const=None, nargs='?',
        help='Number of threads')

    p.add_argument("--doDust", help="dust maps", action="store_true")
    p.add_argument("--doAnimate", help="movie of survey", action="store_true")
    p.add_argument("--doMetrics", help="survey metrics",
                   action="store_true")
    p.add_argument("--doAnimateSkymaps", help="animate skymaps",
                   action="store_true")
    p.add_argument("--doPlotSkymaps", help="plot skymaps", action="store_true")
    p.add_argument("--doSlicer", help="efficiency studies",
                   action="store_true")
    p.add_argument("--doLimitingMagnitudes", help="add limiting magnitudes",
                   action="store_true")
    p.add_argument("--doOuterLoopOnly", help="just the outer loop, no inner",
                   action="store_true")
    p.add_argument("--doLMCSMC", help="LMC/SMC survey",
                   action="store_true")
    p.add_argument("--doDownlink", help="Downlink times",
                   action="store_true")
    p.add_argument("--doParallel", help="enable parallelization",
                   action="store_true")

    return p


def get_observed(latest_time, mission, healpix, schedulenames, prob,
                 centers, tau=60.0):

    cnt = 0
    for schedulename in schedulenames:
        scheduletmp = QTable.read(schedulename, format='ascii.ecsv')
        if len(scheduletmp) == 0:
            continue
        if cnt == 0:
            schedule = scheduletmp
        else:
            schedule = vstack([schedule, scheduletmp])
        cnt = cnt + 1
    if cnt == 0:
        return prob / np.sum(prob)

    for col in schedule.colnames:
        schedule[col].info.indices = []
    schedule.add_index('time')

    idx, _, _ = schedule["center"].match_to_catalog_sky(centers)
    probscale = np.ones(prob.shape)
    for cc, cent in enumerate(centers):
        if np.mod(cc, 100) == 0:
            print('%d/%d' % (cc, len(centers)))

        idy = np.where(idx == cc)[0]
        if len(idy) == 0:
            continue
        exps = schedule.iloc[idy]
        tt = max(exps["time"])

        ipix = mission.fov.footprint_healpix(healpix, cent)
        dt = latest_time - tt
        scale = 1 - np.exp(-dt.jd/tau)
        probscale[ipix] = probscale[ipix] * scale

    prob = prob*probscale
    prob = prob / np.sum(prob)

    return prob


def merge_tables(schedulenames):

    cnt = 0
    for ii, schedulename in enumerate(schedulenames):
        schedule = QTable.read(schedulename, format='ascii.ecsv')
        survey = schedulename.split("/")[-1].split("_")[1]
        fitsfile = schedulename.replace("csv", "fits")
        fitsfile = fitsfile.replace("survey_%s" % survey, "skymap_%s" % survey)
        if len(schedule) == 0:
            continue
        schedule.add_column(survey, name='survey')
        schedule.add_column(fitsfile, name='skymap')
        if cnt == 0:
            scheduleall = schedule
        else:
            scheduleall = vstack([scheduleall, schedule])
        cnt = cnt + 1

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
    orb = mission.orbit

    # check for surveys
    survey_block = False
    cnt = 0
    survey_blocks = {}
    while not survey_block:
        label = f"block{cnt}_surveys"
        weights = f"block{cnt}_weights"
        duration = f"block{cnt}_duration"
        filts = f"block{cnt}_filters"

        if label not in config["simsurvey"]:
            break

        survey_blocks[cnt] = {}
        survey_blocks[cnt]["surveys"] = config["simsurvey"][label].split(",")
        survey_blocks[cnt]["weights"] = [0] + \
            [float(x) for x in config["simsurvey"][weights].split(",")]
        survey_blocks[cnt]["weights_cumsum"] = np.cumsum(
            survey_blocks[cnt]["weights"])
        survey_blocks[cnt]["duration"] = u.Quantity(
            config["simsurvey"][duration])
        survey_blocks[cnt]["filters"] = config["simsurvey"][filts].split(",")

        # Set up grids
        with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
            niter = int(np.round(
                (survey_blocks[cnt]["duration"] /
                 args.duration)).to_value(u.dimensionless_unscaled))
        survey_blocks[cnt]["niter"] = niter

        cnt = cnt + 1

    niter = 0
    iters = [0]
    for ii, key in enumerate(survey_blocks.keys()):
        niter = niter + survey_blocks[key]["niter"]
        if ii == 0:
            duration = survey_blocks[key]["duration"]
        else:
            duration = duration + survey_blocks[key]["duration"]
        iters.append(niter)
    iters_cumsum = np.cumsum(iters)

    assert duration == args.duration_survey

    # Set up pointing grid
    if args.skygrid_file is not None:
        centers = Table.read(args.skygrid_file, format='ascii.ecsv')['center']
    else:
        centers = getattr(skygrid, args.skygrid_method.replace('-', '_'))(
            args.skygrid_step)

    coords = healpix.healpix_to_skycoord(np.arange(healpix.npix))

    outdir = args.output
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    if args.doDust:
        from dustmaps.planck import PlanckQuery
        from ..dust import Dust

        planck = PlanckQuery()
        dust_properties = Dust()

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

    randvals = np.random.rand(niter)

    if args.gw_too_file is not None:
        df = pd.read_csv(args.gw_too_file, delimiter=',')
        gwfits, gwexps = [], []
        for index, row in df.iterrows():
            filename = os.path.join(args.gw_folder,
                                    '%d.fits' % row['coinc_event_id'])
            gwfits.append(filename)
            gwexps.append(float(row['t_exp (ks)'])*1000*u.s)
    else:
        gwfits = glob.glob(os.path.join(args.gw_folder, '*.fits')) +\
            glob.glob(os.path.join(args.gw_folder, '*.fits.fz'))
        gwexps = [args.extime]*len(gwfits)

    if args.doLMCSMC:
        lmc = SkyCoord(ra=80.894200*u.deg, dec=-69.756100*u.deg)
        smc = SkyCoord(ra=13.158300*u.deg, dec=-72.800300*u.deg)

        idx1, _, _ = lmc.match_to_catalog_sky(centers)
        idx2, _, _ = smc.match_to_catalog_sky(centers)
        idx = np.array([idx1, idx2])
        smclmctiles = centers[idx]

    if args.doDownlink:
        with u.add_enabled_equivalencies(equivalencies.orbital(mission.orbit)):
            downlink_times = start_time + np.arange(
                0, args.duration_survey.to_value(u.s),
                (6*u.hr).to_value(u.s)) * u.s

    schedulenames = []
    tind = 0

    for jj in range(niter):
        print('Evaluating iteration: %d/%d' % (jj, niter))

        idy = np.where((iters_cumsum[1:] > jj) & (iters_cumsum[:-1] <= jj))[0]
        key = list(survey_blocks.keys())[int(idy)]

        randval = randvals[jj]
        idx = np.where((survey_blocks[key]["weights_cumsum"][1:] >
                        randval) &
                       (survey_blocks[key]["weights_cumsum"][:-1] <=
                        randval))[0]
        filters = survey_blocks[key]["filters"]

        survey = survey_blocks[key]["surveys"][int(idx)]
        schedulename = '%s/survey_%s_%05d.csv' % (outdir, survey, jj)
        skymapname = '%s/skymap_%s_%05d.fits' % (outdir, survey, jj)
        gifname = '%s/skymap_%s_%05d.gif' % (outdir, survey, jj)

        with u.add_enabled_equivalencies(equivalencies.orbital(orb)):
            times = start_time + args.delay + np.arange(
                0, args.duration.to_value(u.s),
                args.time_step.to_value(u.s)) * u.s

        if survey == "GW":
            idx = int(np.floor(len(gwfits)*np.random.rand()))
            gwskymap = gwfits[idx]
            exptime = gwexps[idx]
            time_step = gwexps[idx]
            delay = 8 * u.hr
        else:
            exptime = args.exptime
            time_step = args.time_step
            delay = 0 * u.hr

        if os.path.isfile(schedulename):
            schedulenames.append(schedulename)

            start_time = times[-1] + exptime
            if survey == "baseline":
                tind = tind + 1
                tind = np.mod(tind, quadlen)
            continue

        if survey == "GW":
            skymap = read_sky_map(gwskymap,
                                  moc=True)['UNIQ', 'PROBDENSITY']
            prob = rasterize(
                skymap, nside_to_level(healpix.nside))['PROB']
            prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]
            if args.doDust:
                prob = prob*V
        elif survey == "dropout":
            prob = 0.00 * np.ones(healpix.npix)
        elif survey in ["galactic_plane", "kilonova", "baseline"]:
            n = 0.01 * np.ones(healpix.npix)
            prob = n / np.sum(n)
            prob = get_observed(start_time, mission, healpix,
                                schedulenames, prob, centers)

            if survey == "galactic_plane":
                p = (np.abs(coords.galactic.b.deg) <= 15.0)
            elif survey == "kilonova":
                tindex = 37
                tquad = quad.loc[tindex]
                raquad, decquad = tquad["center"].ra, tquad["center"].dec
                p = quad_fov.footprint_healpix(healpix,
                                               SkyCoord(raquad, decquad))
            elif survey == "baseline":
                tquad = quad.loc[tind]
                raquad, decquad = tquad["center"].ra, tquad["center"].dec
                p = quad_fov.footprint_healpix(healpix,
                                               SkyCoord(raquad, decquad))
                tind = tind + 1
                tind = np.mod(tind, quadlen)

            prob[p] = 1.
            prob = prob / np.sum(prob)
            prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]
            if args.doDust:
                prob = prob*V

        executable = 'dorado-scheduling'
        write_sky_map(skymapname, prob, moc=True,
                      gps_time=(start_time-delay).gps)

        if args.doPlotSkymaps:
            system_command = 'ligo-skymap-plot %s -o %s' % (
                skymapname, skymapname.replace("fits", "png"))
            os.system(system_command)

        schedulename = '%s/survey_%s_%05d.csv' % (outdir,
                                                  survey,
                                                  jj)

        if survey == "dropout":
            result = QTable(data={'time': [],
                                  'exptime': [],
                                  'location': [],
                                  'center': [],
                                  'roll': []})
            result.write(schedulename, format='ascii.ecsv')

        elif not args.doOuterLoopOnly:
            system_command = ("%s %s -o %s --mission %s --exptime '%s' "
                              "--time-step '%s' --roll-step '%s' "
                              "--skygrid-file %s --duration '%s' "
                              "--timeout %d --delay '%s'") % (
                executable,
                skymapname, schedulename, args.mission,
                str(exptime), str(time_step), str(args.roll_step),
                args.skygrid_file.name, str(args.duration),
                args.timeout, str(delay))
            print(system_command)
            os.system(system_command)
        else:
            skymap = read_sky_map(skymapname,
                                  moc=True)['UNIQ', 'PROBDENSITY']
            prob = rasterize(
                skymap, nside_to_level(healpix.nside))['PROB']
            prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]
            idx = np.argmax(prob)

            theta, phi = hp.pix2ang(healpix.nside, np.arange(len(prob)))
            ra = np.rad2deg(phi)[idx]
            dec = np.rad2deg(0.5*np.pi - theta)[idx]

            result = QTable(data={'time': [start_time],
                                  'exptime': [exptime],
                                  'location': [orb(start_time
                                                   ).earth_location],
                                  'center': [SkyCoord(ra*u.deg,
                                                      dec*u.deg)],
                                  'roll': [0 * u.deg]})
            result.write(schedulename, format='ascii.ecsv')

        with u.add_enabled_equivalencies(equivalencies.orbital(orb)):
            start_time = start_time + args.duration

        schedule_tmp = QTable.read(schedulename, format='ascii.ecsv')
        if args.doLMCSMC and not (survey == "dropout"):
            start_time = times[-1] + exptime

            smclmctimes = start_time + np.arange(
                0, args.exptime.to_value(u.s) * 7,
                args.exptime.to_value(u.s)) * u.s

            check = mission.get_field_of_regard(smclmctiles, smclmctimes,
                                                jobs=args.jobs)
            # I am only adding this if I can do both
            if np.trace(check) == 7:
                result = QTable(data={'time': smclmctimes,
                                      'exptime': [5*args.exptime,
                                                  2*args.exptime],
                                      'location': orb(smclmctimes
                                                      ).earth_location,
                                      'center': smclmctiles,
                                      'roll': [0 * u.deg, 0 * u.deg]})
                if len(schedule_tmp) > 0:
                    schedule_tmp = vstack([schedule_tmp, result])
                else:
                    schedule_tmp = copy.deepcopy(result)
                start_time = smclmctimes[-1] + args.exptime

        for ii, filt in enumerate(filters):
            schedule_tmp_filt = copy.deepcopy(schedule_tmp)
            if len(schedule_tmp_filt) == 0:
                schedule_tmp_filt.add_column([], name='filter')
            else:
                schedule_tmp_filt.add_column(filt, name='filter')
            if ii == 0:
                schedule = schedule_tmp_filt
            else:
                # Don't need empty rows...
                if len(schedule_tmp_filt) == 0:
                    continue
                schedule = vstack([schedule, schedule_tmp_filt])
        schedule.sort('time')
        if args.doDownlink:
            if len(schedule) > 1:
                idy = np.where((schedule[0]["time"] <= downlink_times) &
                               (schedule[1]["time"] > downlink_times))[0]
                for tt in downlink_times[idy]:
                    idx = np.where(schedule["time"] >= tt)[0]
                    if len(idx) == 0:
                        continue
                    schedule["time"] = schedule["time"] + 40*u.min

        if args.doLimitingMagnitudes:
            from uvex.sensitivity import limiting_mag
            if args.doParallel:
                from joblib import Parallel, delayed
                limmags = Parallel(n_jobs=args.jobs)(
                    delayed(limiting_mag)(row['center'],
                                          row['time'],
                                          row['exptime'],
                                          row['filter'])
                    for row in schedule)
            else:
                limmags = []
                for ii, row in enumerate(schedule):
                    obstime, exposure = row['time'], row['exptime']
                    coord, band = row['center'], row['filter']
                    limmag = limiting_mag(coord, obstime,
                                          exposure=exposure, band=band)
                    limmags.append(limmag)
            schedule.add_column(limmags, name='limmag')

        schedule.write(schedulename, format='ascii.ecsv')

        if args.doPlotSkymaps:
            system_command = 'ligo-skymap-plot %s -o %s' % (
                skymapname, skymapname.replace("fits", "png"))
            os.system(system_command)

        schedulenames.append(schedulename)
        start_time = times[-1] + exptime

    scheduleall = merge_tables(schedulenames)
    schedulename = '%s/metrics/survey_all.csv' % (outdir)
    skymapname = '%s/metrics/survey_all.fits' % (outdir)
    gifname = '%s/metrics/survey_all.mp4' % (outdir)
    gifskymapname = '%s/metrics/skymaps_all.mp4' % (outdir)
    metricsname = '%s/metrics' % (outdir)
    if not os.path.isdir(metricsname):
        os.makedirs(metricsname)

    scheduleall.write(schedulename, format='ascii.ecsv')

    n = np.ones(healpix.npix)
    prob = n / np.sum(n)
    write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

    if args.doMetrics:
        executable = 'dorado-scheduling-survey-metrics'
        system_command = '%s %s --mission %s -o %s --skygrid-file %s' % (
            executable, schedulename, args.mission,
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
        system_command = '%s %s --mission %s -o %s --skygrid-file %s' % (
            executable, schedulename, args.mission, metricsname,
            args.skygrid_file.name)
        print(system_command)
        os.system(system_command)

    if args.doAnimateSkymaps:
        executable = 'dorado-scheduling-animate-skymaps'
        system_command = '%s %s -o %s --nside %d' % (
            executable, schedulename, gifskymapname, args.nside)
        print(system_command)
        os.system(system_command)
