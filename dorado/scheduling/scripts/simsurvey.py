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

from ligo.skymap.tool import ArgumentParser, FileType

from astropy.time import Time
from astropy.table import QTable, vstack
from astropy_healpix import nside_to_level
from astropy.coordinates import SkyCoord
from astropy import units as u

from ligo.skymap.io import read_sky_map, write_sky_map
from ligo.skymap.bayestar import rasterize

np.random.seed(0)

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('tiles', metavar='FILE.dat',
                   type=FileType('rb'), help='baseline tiling file')
    p.add_argument('-n', '--norb', default=10,
                   type=int, help='Number of orbit')
    p.add_argument('-s', '--start_time', type=str,
                   default='2020-01-01T00:00:00')
    p.add_argument('--output', '-o',
                   type=str, default='simsurvey',
                   help='output survey')
    p.add_argument('-c', '--config', help='config file')

    p.add_argument("--doAnimateInd", help="load CSV", action="store_true")
    p.add_argument("--doAnimateAll", help="load CSV", action="store_true")
    p.add_argument("--doMetrics", help="load CSV", action="store_true")
    p.add_argument("--doPlotSkymaps", help="load CSV", action="store_true")

    return p


def getSquarePixels(ra_pointing, dec_pointing, tileSide, nside):

    decCorners = (dec_pointing - tileSide / 2.0, dec_pointing + tileSide / 2.0)

    # security for the periodic limit conditions
    radecs = []
    for d in decCorners:
        if d > 90.:
            d = 180. - d
        elif d < -90.:
            d = -180 - d

        raCorners = (ra_pointing - (tileSide / 2.0) / np.cos(np.deg2rad(d)),
                     ra_pointing + (tileSide / 2.0) / np.cos(np.deg2rad(d)))

        # security for the periodic limit conditions
        for r in raCorners:
            if r > 360.:
                r = r - 360.
            elif r < 0.:
                r = 360. + r
            radecs.append([r, d])

    radecs = np.array(radecs)
    idx1 = np.where(np.abs(radecs[:, 1]) >= 87.0)[0]
    if len(idx1) == 4:
        return []

    idx1 = np.where((radecs[:, 1] >= 87.0) | (radecs[:, 1] <= -87.0))[0]
    if len(idx1) > 0:
        radecs = np.delete(radecs, idx1[0], 0)

    xyz = []
    for r, d in radecs:
        xyz.append(hp.ang2vec(r, d, lonlat=True))

    npts, junk = radecs.shape
    if npts == 4:
        xyz = [xyz[0], xyz[1], xyz[3], xyz[2]]
        try:
            ipix = hp.query_polygon(nside, np.array(xyz))
        except Exception:
            return []
    else:
        ipix = hp.query_polygon(nside, np.array(xyz))

    return ipix


def get_observed(latest_time, tiling_model, schedulenames, prob):

    ras, decs, tts = [], [], []
    for schedulename in schedulenames:
        schedule = QTable.read(schedulename, format='ascii.ecsv')
        for row in schedule:
            ras.append(row["center"].ra.deg)
            decs.append(row["center"].dec.deg)
            tts.append(row["time"])

    probscale = np.ones(prob.shape)
    for ra, dec, tt in zip(ras, decs, tts):
        ipix = getSquarePixels(ra, dec, 3.3, tiling_model.healpix.nside)
        dt = latest_time - tt
        tau = 60.0
        scale = 1 - np.exp(-dt.jd/tau)
        probscale[ipix] = probscale[ipix] * scale

    prob = prob*probscale
    prob = prob / np.sum(prob)

    return prob


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
    from ..models import TilingModel

    if args.config is not None:
        config = configparser.ConfigParser()
        config.read(args.config)
        satfile = config["survey"]["satfile"]
        exposure_time = float(config["survey"]["exposure_time"]) * u.minute
        steps_per_exposure =\
            int(config["survey"]["time_steps_per_exposure"])
        field_of_view = float(config["survey"]["field_of_view"]) * u.deg
        tiling_model = TilingModel(satfile=satfile,
                                   exposure_time=exposure_time,
                                   time_steps_per_exposure=steps_per_exposure,
                                   field_of_view=field_of_view)
    else:
        tiling_model = TilingModel()

    npix = tiling_model.healpix.npix
    nside = tiling_model.healpix.nside

    outdir = args.output
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    tess = np.loadtxt(args.tiles)
    tess = tess[8:, :]
    tesslen = len(tess)

    start_time = Time(args.start_time, format='isot')

    exposure_time = tiling_model.exposure_time

    surveys = ['baseline', 'galactic_plane', 'kilonova', 'GW']
    weights = [0, 0.5, 0.2, 0.2, 0.1]
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
            times = np.arange(tiling_model.time_steps) *\
                tiling_model.time_step_duration + start_time
            start_time = times[-1] + exposure_time
            tind = np.mod(tind, tesslen)
            continue

        if survey == "galactic_plane":
            theta, phi = hp.pix2ang(nside, np.arange(npix))
            ra = np.rad2deg(phi)
            dec = np.rad2deg(0.5*np.pi - theta)
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            idx = np.where(np.abs(coords.galactic.b.deg) <= 15.0)[0]
            n = 0.01 * np.ones(npix)
            n[idx] = 1.0
            prob = n / np.sum(n)

            prob = get_observed(start_time, tiling_model, schedulenames, prob)

        elif survey == "kilonova":
            n = 0.01 * np.ones(npix)

            field = tess[143, :]
            p = getSquarePixels(field[1], field[2], 25.0, nside)
            n[p] = 1.
            prob = n / np.sum(n)

        elif survey == "GW":
            idx = int(np.floor(10*np.random.rand()))
            gwskymap = 'GW/%d.fits' % idx
            skymap = read_sky_map(gwskymap, moc=True)['UNIQ', 'PROBDENSITY']
            prob = rasterize(skymap, nside_to_level(nside))['PROB']
            prob = prob[tiling_model.healpix.ring_to_nested(np.arange(
                                                            len(prob)))]

        elif survey == "baseline":
            n = 0.01 * np.ones(npix)

            field = tess[tind, :]
            tind = tind + 1
            tind = np.mod(tind, tesslen)

            p = getSquarePixels(field[1], field[2], 25.0, nside)
            n[p] = 1.
            prob = n / np.sum(n)

            prob = get_observed(start_time, tiling_model, schedulenames, prob)

        write_sky_map(skymapname, prob, moc=True, gps_time=start_time.gps)

        times = np.arange(tiling_model.time_steps) *\
            tiling_model.time_step_duration + start_time

        executable = 'dorado-scheduling-survey'
        system_command = '%s %s examples/tiles.dat -o %s -s %s' % (
            executable, skymapname, schedulename, start_time.isot)
        print(system_command)
        os.system(system_command)

        if args.doAnimateInd:
            executable = 'dorado-scheduling-animate-survey'
            system_command = '%s %s %s %s -s %s' % (
                executable, skymapname, schedulename, gifname, start_time.isot)
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
        system_command = 'dorado-scheduling-survey-metrics %s %s %s' % (
            skymapname, schedulename, metricsname)
        print(system_command)
        os.system(system_command)

    if args.doAnimateAll:
        start_time = scheduleall[0]["time"]
        system_command = 'dorado-scheduling-animate-survey %s %s %s -s %s' % (
            skymapname, schedulename, gifname, start_time.isot)
        print(system_command)
        os.system(system_command)
