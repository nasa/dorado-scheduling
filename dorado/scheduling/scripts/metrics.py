#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Plot an observing plan."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

from .. import mission as _mission
from .. import skygrid

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling-metrics')
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

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        '--skygrid-method', default='healpix',
        choices=[key.replace('_', '-') for key in skygrid.__all__],
        help='Sky grid method')
    group.add_argument(
        '--skygrid-file', metavar='TILES.ecsv',
        type=FileType('rb'),
        help='tiles filename')

    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    from astropy.table import Table
    from ligo.skymap import plot
    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.pyplot import cm
    import numpy as np
    import seaborn

    mission = getattr(_mission, args.mission)
    tiles = Table.read(args.skygrid_file, format='ascii.ecsv')
    centers = tiles['center']

    outdir = args.output
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    log.info('reading observing schedule')
    schedule = Table.read(args.schedule.name, format='ascii.ecsv')
    for col in schedule.colnames:
        schedule[col].info.indices = []
    schedule.add_index('time')

    idx, _, _ = schedule["center"].match_to_catalog_sky(centers)
    survey_set = list(set(schedule["survey"]))
    colors = seaborn.color_palette('Set2', n_colors=len(survey_set))

    dts = {}
    exposures = {}
    for survey in survey_set:
        dts[survey] = []
        exposures[survey] = np.zeros((len(centers), 1))
    exposures['all'] = np.zeros((len(centers), 1))
    for cc, cent in enumerate(centers):
        idy = np.where(idx == cc)[0]
        if len(idy) == 0:
            continue
        exps = schedule.iloc[idy]
        exposures['all'][cc] = len(exps)
        for ss, survey in enumerate(survey_set):
            idz = np.where(exps["survey"] == survey)[0]
            if len(idz) == 0:
                continue
            exposures[survey][cc] = len(idz)
            for ii in range(len(idz)-1):
                jj, kk = idz[ii], idz[ii+1]
                dt = exps[kk]["time"] - exps[jj]["time"]
                if np.isclose(dt.jd, 0.0):
                    continue
                dts[survey].append(dt.jd)

    fig = plt.figure(figsize=(8, 6))
    bin_edges = np.linspace(0, 60, 61)
    for ii, survey in enumerate(survey_set):
        hist, _ = np.histogram(dts[survey], bins=bin_edges)
        bins = (bin_edges[1:]+bin_edges[:-1])/2.0
        plt.step(bins, hist, color=colors[ii], linestyle='--', label=survey)
    plt.xlabel('Time between observations [days]')
    plt.ylabel('Counts')
    plt.legend(loc=1)
    plotname = os.path.join(outdir, 'dt.pdf')
    plt.savefig(plotname)
    plt.close()

    colors = cm.rainbow(np.linspace(0, 1, 10))
    vmin, vmax = 0, 30
    colorbar = np.linspace(vmin, vmax, len(colors))
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    for survey in survey_set + ["all"]:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes([0.05, 0.05, 0.85, 0.9],
                      projection='astro hours mollweide')
        ax.grid()
        for cc, center in enumerate(centers):
            poly = mission.fov.footprint(center).icrs
            idx = np.argmin(np.abs(colorbar - exposures[survey][cc]))
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
        cbar.set_label(r'Number of Observations')
        plotname = os.path.join(outdir, 'counts_%s.pdf' % survey)
        plt.savefig(plotname)
        plt.close()


if __name__ == '__main__':
    main()
