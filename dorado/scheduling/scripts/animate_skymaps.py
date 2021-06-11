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

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser(prog='dorado-scheduling-animate-survey')
    p.add_argument('schedule', metavar='SCHEDULE.ecsv',
                   type=FileType('rb'), default='-',
                   help='Schedule filename')
    p.add_argument('--output', '-o',
                   metavar='MOVIE.gif', type=FileType('wb'),
                   help='Output filename')
    p.add_argument(
        '--nside', type=int, default=32, help='HEALPix sampling resolution')

    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    from astropy_healpix import HEALPix, nside_to_level, npix_to_nside
    from astropy.coordinates import ICRS
    from astropy.table import QTable
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.postprocess import find_greedy_credible_levels
    from matplotlib import pyplot as plt
    from matplotlib.animation import FuncAnimation
    import numpy as np
    from tqdm import tqdm

    healpix = HEALPix(args.nside, order='nested', frame=ICRS())

    log.info('reading observing schedule')
    schedule = QTable.read(args.schedule.name, format='ascii.ecsv')

    log.info('reading skymaps')
    clss = []
    skymaps = []
    skymapfiles = []
    for ii, row in enumerate(schedule):
        if np.mod(ii, 100) == 0:
            print('%d/%d' % (ii, len(schedule)))

        if row['skymap'] not in skymapfiles:
            skymap = read_sky_map(row['skymap'],
                                  moc=True)['UNIQ', 'PROBDENSITY']
            skymap_hires = rasterize(skymap)['PROB']
            healpix_hires = HEALPix(npix_to_nside(len(skymap_hires)))
            skymap = rasterize(skymap,
                               nside_to_level(args.nside))
            skymap = skymap['PROB']
            nest = healpix.order == 'nested'
            if not nest:
                skymap = skymap[healpix.ring_to_nested(np.arange(
                    len(skymap)))]
                skymap_hires = skymap[healpix_hires.ring_to_nested(np.arange(
                    len(skymap_hires)))]
            cls = find_greedy_credible_levels(skymap_hires)
            skymaps.append(skymap)
            clss.append(cls)
            skymapfiles.append(row['skymap'])

    fig = plt.figure(figsize=(8, 8))

    ax_sky = fig.add_subplot(projection='astro hours mollweide')
    ax_sky.grid()

    old_artists = []

    log.info('rendering animation frames')
    with tqdm(total=len(skymaps)) as progress:

        def animate(i):
            for artist in old_artists:
                artist.remove()
            del old_artists[:]
            old_artists.append(ax_sky.imshow_hpx((skymaps[i], 'ICRS'),
                                                 nested=nest,
                                                 cmap='cylon'))

            progress.update()

        frames = [ii for ii in range(len(skymaps))]

        ani = FuncAnimation(fig, animate, frames=frames)
        # ani.save(args.output.name, writer=PillowWriter())
        ani.save(args.output.name, fps=1, extra_args=['-vcodec', 'libx264'])
        fig.savefig(args.output.name.replace("mp4", "pdf"))


if __name__ == '__main__':
    main()
