#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Evaluate metrics for an observing plan."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

from .. import mission as _mission

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling-survey-slicer')
    p.add_argument('config', help='config file')
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('schedule', metavar='SCHEDULE.ecsv',
                   type=FileType('rb'), default='-',
                   help='Schedule filename')

    p.add_argument('--mission', choices=set(_mission.__all__) - {'Mission'},
                   default='dorado', help='Mission configuration')
    p.add_argument('--output', '-o',
                   type=str, default='simsurvey/metrics',
                   help='output survey')
    p.add_argument(
        '--nside', type=int, default=32, help='HEALPix sampling resolution')

    return p


def main(args=None):
    args = parser().parse_args(args)

    import configparser
    import os
    import numpy as np
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap import plot
    from astropy_healpix import HEALPix, nside_to_level
    from astropy.coordinates import ICRS
    from astropy.table import QTable
    from astropy import units as u
    from astropy.time import Time
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm

    from ..metrics.kne import KNePopMetric, generateKNPopSlicer

    config = configparser.ConfigParser()
    config.read(args.config)

    mission = getattr(_mission, args.mission)
    healpix = HEALPix(args.nside, order='nested', frame=ICRS())

    output = args.output
    if not os.path.isdir(output):
        os.makedirs(output)

    log.info('reading sky map')
    # Read multi-order sky map and rasterize to working resolution
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap,
                     nside_to_level(healpix.nside))['PROB']
    if healpix.order == 'ring':
        prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]

    log.info('reading observing schedule')
    schedule = QTable.read(args.schedule.name, format='ascii.ecsv')

    filtslist = config["filters"]["filters"].split(",")
    magslist = [float(x) for x in config["filters"]["limmags"].split(",")]
    weights = [0] + [float(x) for x in
                     config["filters"]["weights"].split(",")]
    weights_cumsum = np.cumsum(weights)

    limmags, filts = [], []
    randvals = np.random.rand(len(schedule))
    for jj in range(len(schedule)):
        randval = randvals[jj]
        idx = int(np.where((weights_cumsum[1:] >= randval) &
                           (weights_cumsum[:-1] <= randval))[0][0])

        limmags.append(magslist[idx])
        filts.append(filtslist[idx])
    schedule.add_column(limmags, name='limmag')
    schedule.add_column(filts, name='filter')

    times = schedule["time"]

    t = (times - times[0]).to(u.minute).value

    # Generate the slicer which puts 10,000 events at
    # random spots on the sphere
    n_files = 100
    log.info('generating injections')
    slicer = generateKNPopSlicer(seed=42, n_events=10000, n_files=n_files,
                                 t_start=np.min(times.jd),
                                 t_end=np.max(times.jd))

    log.info('splitting schedule')
    centerstrs = []
    centers_set = []
    centerstrs_set = []
    surveys = []
    centid = []
    for row in schedule:
        cent = row["center"]
        centerstr = "%.5f_%.5f" % (cent.ra.deg, cent.dec.deg)
        if centerstr not in centerstrs:
            centers_set.append(cent)
            centerstrs_set.append(centerstr)
        centerstrs.append(centerstr)
        surveys.append(row["survey"])
        centid.append(centerstrs_set.index(centerstr))
    schedule["centid"] = centid

    mejs = 10**np.random.uniform(-3, -1, n_files)
    vejs = np.random.uniform(0.05, 0.30, n_files)
    betas = np.random.uniform(1.0, 5.0, n_files)
    kappas = 10**np.random.uniform(-1.0, 2.0, n_files)

    log.info('generating metric')
    metric = KNePopMetric(mejs, vejs, betas, kappas)
    metrics_list = ['single_detect', 'multi_detect', 'multi_color_detect']
    detections, exposures, efficiency = {}, {}, {}
    for m in metrics_list:
        detections[m] = np.zeros((len(centerstrs_set),))
        exposures[m] = np.zeros((len(centerstrs_set),))
        efficiency[m] = np.zeros((len(centerstrs_set),))

    filename = os.path.join(output, 'counts.dat')
    fid = open(filename, 'w')
    for cc, centerstr in enumerate(centerstrs_set):
        if np.mod(cc, 100) == 0:
            print('Running %d/%d' % (cc, len(centerstrs_set)))

        kk = np.where(schedule["centid"] == cc)[0]
        print('%d %d' % (cc, len(kk)), file=fid, flush=True)
    fid.close()

    filename = os.path.join(output, 'eff.dat')
    fid = open(filename, 'w')

    for cc, centerstr in enumerate(centerstrs_set):
        if np.mod(cc, 100) == 0:
            print('Running %d/%d' % (cc, len(centerstrs_set)))
        if cc == 2225:
            continue

        kk = np.where(schedule["centid"] == cc)[0]
        if len(kk) == 0:
            continue
        if len(kk) < 2000:
            continue
        for slicePoint in slicer:
            result = metric.run(schedule[kk], slicePoint=slicePoint)
            for m in metrics_list:
                detections[m][cc] = detections[m][cc] + result[m]
                exposures[m][cc] = exposures[m][cc] + 1
        for m in metrics_list:
            efficiency[m][cc] = detections[m][cc] / exposures[m][cc]
        print(cc, len(kk), m, efficiency[m][cc])

        print('%d %.10f %.10f %.10f' % (cc,
                                        efficiency['single_detect'][cc],
                                        efficiency['multi_detect'][cc],
                                        efficiency['multi_color_detect'][cc]),
              file=fid, flush=True)
    fid.close()

    log.info('plotting efficiency')

    colors = cm.rainbow(np.linspace(0, 1, 100))
    for m in metrics_list:
        vmin, vmax = 0, np.max(efficiency[m])
        colorbar = np.linspace(vmin, vmax, len(colors))
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes([0.05, 0.05, 0.85, 0.9],
                      projection='astro hours mollweide')
        ax.grid()
        for cc, center in enumerate(centers_set):
            poly = mission.fov.footprint(center).icrs
            idx = np.argmin(np.abs(colorbar - efficiency[m][cc]))
            footprint_color = colors[idx]
            vertices = np.column_stack((poly.ra.rad, poly.dec.rad))
            for cut_vertices in plot.cut_prime_meridian(vertices):
                patch = plt.Polygon(np.rad2deg(cut_vertices),
                                    transform=ax.get_transform('world'),
                                    facecolor=footprint_color,
                                    edgecolor=footprint_color,
                                    alpha=0.5)
                ax.add_patch(patch)
        cax = ax.inset_axes([0.97, 0.2, 0.05, 0.6], transform=ax.transAxes)
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cm.rainbow),
                            cax=cax)
        cbar.set_label(r'Efficiency')
        plotname = os.path.join(output, 'eff_%s.pdf' % m)
        plt.savefig(plotname)
        plt.close()

    log.info('plotting lightcurves')

    # let's plot up a few of the lightcurves
    ivals = [0, 1, 2, 3, 4, 5]

    plt.figure()
    for i in ivals:
        t = Time(np.arange(0, 10), format='jd')
        distmod = 5*np.log10(slicer[i]['distance']*1e6) - 5.0
        lc = metric.lightcurves.interp(t, 'FUV',
                                       lc_indx=slicer[i]['file_indx'])
        plt.plot(t.jd, lc+distmod, color='blue', label='%i, FUV' % i)
        lc = metric.lightcurves.interp(t, 'NUV',
                                       lc_indx=slicer[i]['file_indx'])
        plt.plot(t.jd, lc+distmod, color='red', label='%i, NUV' % i)
    plt.ylim([30, 20])
    plt.xlabel('t (days)')
    plt.ylabel('mag')
    plt.legend()
    plotname = os.path.join(output, 'lc.pdf')
    plt.savefig(plotname)
    plt.close()
