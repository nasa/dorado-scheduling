#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Simulate full survey."""

import os
import logging
import numpy as np
import healpy as hp

from ligo.skymap.tool import ArgumentParser

from astropy.time import Time
from astropy.table import QTable, vstack
from astropy_healpix import nside_to_level
from astropy.coordinates import SkyCoord
from astropy import units as u

from ligo.skymap.io import read_sky_map, write_sky_map
from ligo.skymap.bayestar import rasterize

from .. import mission as _mission

np.random.seed(0)

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('config', help='config file')
    p.add_argument('mission', choices=set(_mission.__all__) - {'Mission'},
                   default='dorado', help='Mission configuration')
    p.add_argument('-n', '--norb', default=10,
                   type=int, help='Number of orbit')
    p.add_argument('-s', '--start_time', type=str,
                   default='2020-01-01T00:00:00')
    p.add_argument('--output', '-o',
                   type=str, default='simsurvey',
                   help='output survey')
    p.add_argument('--timeout', type=int,
                   default=300, help='Impose timeout on solutions')

    p.add_argument("--doDust", help="load CSV", action="store_true")
    p.add_argument("--doAnimateInd", help="load CSV", action="store_true")
    p.add_argument("--doAnimateAll", help="load CSV", action="store_true")
    p.add_argument("--doMetrics", help="load CSV", action="store_true")
    p.add_argument("--doPlotSkymaps", help="load CSV", action="store_true")
    p.add_argument("--doSlicer", help="load CSV", action="store_true")
    p.add_argument("--doOverlap", help="load CSV", action="store_true")

    return p


def get_observed(latest_time, survey_model, schedulenames, prob):

    ras, decs, tts = [], [], []
    for schedulename in schedulenames:
        schedule = QTable.read(schedulename, format='ascii.ecsv')
        for row in schedule:
            ras.append(row["center"].ra.deg)
            decs.append(row["center"].dec.deg)
            tts.append(row["time"])

    probscale = np.ones(prob.shape)
    for ra, dec, tt in zip(ras, decs, tts):
        ipix = survey_model.mission.fov.footprint_healpix(survey_model.healpix,
                                                          SkyCoord(ra*u.deg,
                                                                   dec*u.deg))
        dt = latest_time - tt
        tau = 60.0
        scale = 1 - np.exp(-dt.jd/tau)
        probscale[ipix] = probscale[ipix] * scale

    prob = prob*probscale
    prob = prob / np.sum(prob)

    return prob


def compute_overlap(survey_model):

    res = hp.nside2resol(survey_model.healpix.nside, arcmin=True)
    ipix = {}
    for ii, cent1 in enumerate(survey_model.centers):
        ra, dec = cent1.ra.deg, cent1.dec.deg
        fov = survey_model.mission.fov
        ipix[ii] = fov.footprint_healpix(survey_model.healpix,
                                         SkyCoord(ra*u.deg, dec*u.deg))
    overlaps = []
    for ii, cent1 in enumerate(survey_model.centers):
        if ii >= 100:
            continue
        overlap = 0.0
        for jj, cent2 in enumerate(survey_model.centers):
            if ii <= jj:
                continue
            over = np.intersect1d(ipix[ii], ipix[jj])
            overlap = np.max([overlap, len(over)*res])
        overlaps.append(overlap)
    print('max overlap: %.1f arcmin' % (np.max(overlaps)))


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
    from ..models import SurveyModel
    from ..fov import FOV

    config = configparser.ConfigParser()
    config.read(args.config)

    mission = getattr(_mission, args.mission)
    tiles = QTable.read(config["survey"]["tilesfile"], format='ascii.ecsv')

    exposure_time = float(config["survey"]["exposure_time"]) * u.minute
    steps_per_exposure =\
        int(config["survey"]["time_steps_per_exposure"])
    exposure_time_steps = exposure_time / steps_per_exposure
    number_of_orbits = int(config["survey"]["number_of_orbits"])
    survey_model = SurveyModel(mission=mission,
                               exposure_time=exposure_time,
                               time_steps_per_exposure=steps_per_exposure,
                               number_of_orbits=number_of_orbits,
                               centers=tiles["center"])

    npix = survey_model.healpix.npix
    nside = survey_model.healpix.nside

    theta, phi = hp.pix2ang(nside, np.arange(npix))
    ra = np.rad2deg(phi)
    dec = np.rad2deg(0.5*np.pi - theta)
    coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

    if args.doOverlap:
        compute_overlap(survey_model)

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

    exposure_time = survey_model.exposure_time
    surveys = config["simsurvey"]["surveys"].split(",")
    weights = [0] + [float(x) for x in
                     config["simsurvey"]["weights"].split(",")]
    weights_cumsum = np.cumsum(weights)

    randvals = np.random.rand(args.norb)

    schedulenames = []
    tind = 0

    for jj in range(args.norb):

        randval = randvals[jj]
        idx = np.where((weights_cumsum[1:] >= randval) &
                       (weights_cumsum[:-1] <= randval))[0]

        survey = surveys[int(idx)]
        schedulename = '%s/survey_%s_%05d.csv' % (outdir, survey, jj)
        skymapname = '%s/skymap_%s_%05d.fits' % (outdir, survey, jj)
        gifname = '%s/skymap_%s_%05d.gif' % (outdir, survey, jj)

        if os.path.isfile(schedulename):
            schedulenames.append(schedulename)
            times = np.arange(survey_model.time_steps) *\
                survey_model.time_step_duration + start_time
            start_time = times[-1] + exposure_time
            tind = tind + 1
            tind = np.mod(tind, quadlen)
            continue

        if survey == "galactic_plane":
            idx = np.where(np.abs(coords.galactic.b.deg) <= 15.0)[0]
            n = 0.01 * np.ones(npix)
            n[idx] = 1.0
            prob = n / np.sum(n)

            prob = get_observed(start_time, survey_model, schedulenames, prob)
            if args.doDust:
                prob = prob*V

        elif survey == "kilonova":
            n = 0.01 * np.ones(npix)

            tindex = int(quadlen/2)
            tquad = quad.loc[tindex]
            raquad, decquad = tquad["center"].ra, tquad["center"].dec
            p = quad_fov.footprint_healpix(survey_model.healpix,
                                           SkyCoord(raquad, decquad))
            n[p] = 1.
            prob = n / np.sum(n)
            if args.doDust:
                prob = prob*V

        elif survey == "GW":
            idx = int(np.floor(10*np.random.rand()))
            gwskymap = 'GW/%d.fits' % idx
            skymap = read_sky_map(gwskymap, moc=True)['UNIQ', 'PROBDENSITY']
            prob = rasterize(skymap, nside_to_level(nside))['PROB']
            prob = prob[survey_model.healpix.ring_to_nested(np.arange(
                                                            len(prob)))]
            if args.doDust:
                prob = prob*V

        elif survey == "baseline":
            n = 0.01 * np.ones(npix)

            tquad = quad.loc[tind]
            raquad, decquad = tquad["center"].ra, tquad["center"].dec
            p = quad_fov.footprint_healpix(survey_model.healpix,
                                           SkyCoord(raquad, decquad))
            n[p] = 1.
            prob = n / np.sum(n)
            prob = get_observed(start_time, survey_model, schedulenames, prob)
            if args.doDust:
                prob = prob*V

            tind = tind + 1
            tind = np.mod(tind, quadlen)

        write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

        times = np.arange(survey_model.time_steps) *\
            survey_model.time_step_duration + start_time

        executable = 'dorado-scheduling'
        system_command = ("%s %s -o %s --mission %s --exptime '%s' "
                          "--time-step '%s' --roll-step '90 deg' "
                          "--skygrid-file %s --duration '%d orbit' "
                          "--timeout %d") % (
            executable,
            skymapname, schedulename, args.mission,
            str(exposure_time), str(exposure_time_steps),
            config["survey"]["tilesfile"], number_of_orbits,
            args.timeout)
        print(system_command)
        os.system(system_command)

        if args.doAnimateInd:
            executable = 'dorado-scheduling-animate-survey'
            system_command = '%s %s %s %s %s %s -s %s' % (
                executable, skymapname, args.config, args.mission,
                schedulename, gifname, start_time.isot)
            os.system(system_command)

        if args.doPlotSkymaps:
            system_command = 'ligo-skymap-plot %s -o %s' % (
                skymapname, skymapname.replace("fits", "png"))
            os.system(system_command)

        schedulenames.append(schedulename)
        start_time = times[-1] + exposure_time

    scheduleall = merge_tables(schedulenames)
    schedulename = '%s/metrics/survey_all.csv' % (outdir)
    skymapname = '%s/metrics/survey_all.fits' % (outdir)
    gifname = '%s/metrics/survey_all.mp4' % (outdir)
    metricsname = '%s/metrics' % (outdir)
    if not os.path.isdir(metricsname):
        os.makedirs(metricsname)

    scheduleall.write(schedulename, format='ascii.ecsv')

    n = np.ones((npix,))
    prob = n / np.sum(n)
    write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

    if args.doMetrics:
        executable = 'dorado-scheduling-survey-metrics'
        system_command = '%s %s %s %s %s %s' % (
            executable, skymapname, args.config, args.mission,
            schedulename, metricsname)
        print(system_command)
        os.system(system_command)

    if args.doAnimateAll:
        start_time = scheduleall[0]["time"]
        executable = 'dorado-scheduling-animate-survey'
        system_command = '%s %s %s %s %s %s -s %s' % (
            executable, skymapname, args.config, args.mission,
            schedulename, gifname, start_time.isot)
        print(system_command)
        os.system(system_command)

    if args.doSlicer:
        executable = 'dorado-scheduling-survey-slicer'
        system_command = '%s %s %s %s %s %s' % (
            executable, skymapname, args.config, args.mission,
            schedulename, metricsname)
        print(system_command)
        os.system(system_command)
