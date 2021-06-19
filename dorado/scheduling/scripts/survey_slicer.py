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
from .. import skygrid

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling-survey-slicer')
    p.add_argument('schedule', metavar='SCHEDULE.ecsv',
                   type=FileType('rb'), default='-',
                   help='Schedule filename')

    p.add_argument('--mission', choices=set(_mission.__all__) - {'Mission'},
                   default='dorado', help='Mission configuration')
    p.add_argument('--output', '-o',
                   type=str, default='simsurvey/metrics',
                   help='output survey')

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--skygrid-method', default='healpix',
        choices=[key.replace('_', '-') for key in skygrid.__all__],
        help='Sky grid method')
    group.add_argument(
        '--skygrid-file', metavar='TILES.ecsv',
        type=FileType('rb'),
        help='tiles filename')

    p.add_argument(
        '-j', '--jobs', type=int, default=1, const=None, nargs='?',
        help='Number of threads')

    p.add_argument(
        '-n', '--ninj', type=int, default=10000,
        help='Number of injections')

    p.add_argument(
        '-f', '--field-number', type=int,
        help='field number')

    p.add_argument("--doParallel", help="enable parallelization",
                   action="store_true")
    p.add_argument("--doPlots", help="make plot", action="store_true")

    return p


def main(args=None):
    args = parser().parse_args(args)

    import os
    import numpy as np
    from ligo.skymap import plot
    from astropy.table import QTable
    from astropy import units as u
    from astropy.time import Time
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import cm
    from dustmaps.planck import PlanckQuery

    from ..metrics.kne import KNePopMetric, generateKNPopSlicer
    from ..dust import Dust

    mission = getattr(_mission, args.mission)
    tiles = QTable.read(args.skygrid_file, format='ascii.ecsv')
    centers = tiles['center']

    output = args.output
    if not os.path.isdir(output):
        os.makedirs(output)

    log.info('reading observing schedule')
    schedule = QTable.read(args.schedule.name, format='ascii.ecsv')
    for col in schedule.colnames:
        schedule[col].info.indices = []
    schedule.add_index('time')

    times = schedule["time"]

    t = (times - times[0]).to(u.minute).value

    # Generate the slicer which puts 10,000 events at
    # random spots on the sphere
    n_files = 100
    log.info('generating injections')
    slicer = generateKNPopSlicer(seed=42, n_events=args.ninj,
                                 n_files=n_files,
                                 t_start=np.min(times.jd),
                                 t_end=np.max(times.jd))

    idx, _, _ = schedule["center"].match_to_catalog_sky(centers)

    log.info('splitting schedule')
    if args.field_number is not None:
        filename = os.path.join(output, 'counts_%05d.dat' % args.field_number)
    else:
        filename = os.path.join(output, 'counts.dat')
    fid = open(filename, 'w')

    exposures = {}
    for cc, cent in enumerate(centers):
        if args.field_number is not None:
            if not cc == args.field_number:
                continue

        if np.mod(cc, 100) == 0:
            print('splitting %d/%d' % (cc, len(centers)))

        idy = np.where(idx == cc)[0]
        if len(idy) == 0:
            continue
        exps = schedule.iloc[idy]
        exposures[cc] = exps

        print('%d %d' % (cc, len(exps)), file=fid, flush=True)
    fid.close()

    mejs = 10**np.random.uniform(-3, -1, n_files)
    vejs = np.random.uniform(0.05, 0.30, n_files)
    betas = np.random.uniform(1.0, 5.0, n_files)
    kappas = 10**np.random.uniform(-1.0, 2.0, n_files)

    log.info('generating metric')
    metric = KNePopMetric(mejs, vejs, betas, kappas)
    metrics_list = ['single_detect', 'multi_detect', 'multi_color_detect']
    detections, injections, efficiency = {}, {}, {}
    for m in metrics_list:
        detections[m] = np.zeros((len(centers),))
        injections[m] = np.zeros((len(centers),))
        efficiency[m] = np.zeros((len(centers),))

    planck = PlanckQuery()
    dust_properties = Dust()
    Ax1 = dust_properties.Ax1
    ebv = planck(centers)

    for cc, e in zip(exposures.keys(), ebv):
        if args.field_number is not None:
            if not cc == args.field_number:
                continue

        filename = os.path.join(output, 'eff_%05d.dat' % cc)
        if os.path.isfile(filename):
            continue

        fid = open(filename, 'w')

        if np.mod(int(cc), 100) == 0:
            print('Running %d/%d' % (cc, len(centers)))

        exps = exposures[cc]
        if len(exps) == 0:
            print('%d %.10f %.10f %.10f' % (cc, 0.0, 0.0, 0.0),
                  file=fid, flush=True)
            fid.close()
            continue

        # Apply dust extinction on the light curve
        extinction = {}
        for filt in Ax1.keys():
            extinction[filt] = Ax1[filt] * e

        if args.doParallel:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=args.jobs)(
                delayed(metric.run)(exps,
                                    slicePoint,
                                    extinction)
                for slicePoint in slicer)
        else:
            results = []
            for slicePoint in slicer:
                result = metric.run(exps, slicePoint=slicePoint,
                                    extinction=extinction)
                results.append(result)
        for result in results:
            for m in metrics_list:
                detections[m][cc] = detections[m][cc] + result[m]
                injections[m][cc] = injections[m][cc] + 1
        for m in metrics_list:
            efficiency[m][cc] = detections[m][cc] / injections[m][cc]

        print('%d %.10f %.10f %.10f' % (cc,
                                        efficiency['single_detect'][cc],
                                        efficiency['multi_detect'][cc],
                                        efficiency['multi_color_detect'][cc]),
              file=fid, flush=True)
        fid.close()

    for cc, e in zip(exposures.keys(), ebv):
        filename = os.path.join(output, 'eff_%05d.dat' % cc)
        if not os.path.isfile(filename):
            continue
        data_out = np.loadtxt(filename)
        efficiency['single_detect'][cc] = data_out[1]
        efficiency['multi_detect'][cc] = data_out[2]
        efficiency['multi_color_detect'][cc] = data_out[3]

    if args.doPlots:

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
            for cc, center in enumerate(centers):
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
            plotname = os.path.join(output, 'eff_%s.png' % m)
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
