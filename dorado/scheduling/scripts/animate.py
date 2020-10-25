"""Plot an observing plan."""
import logging

from astropy_healpix import nside_to_level
from astropy.io import fits
from astropy.time import Time
# from astropy.table import Table
from astropy import units as u
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from ligo.skymap import plot
from ligo.skymap.tool import ArgumentParser, FileType
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn
from tqdm import tqdm

from .. import orbit
from .. import skygrid
from ..regard import get_field_of_regard

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    # p.add_argument('schedule', metavar='SCHEDULE.ecsv',
    #                type=FileType('r'), default='-',
    #                help='Schedule filename')
    p.add_argument('output', metavar='MOVIE.mp4', type=FileType('wb'),
                   help='Output filename')
    return p


def main(args=None):
    args = parser().parse_args(args)

    log.info('reading sky map')

    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)
    skymap['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, nside_to_level(skygrid.healpix.nside))['PROB']
    nest = skygrid.healpix.order == 'nested'
    if not nest:
        prob = prob[skygrid.healpix.ring_to_nested(np.arange(len(prob)))]

    times = np.arange(orbit.time_steps) * orbit.exposure_time / orbit.time_steps_per_exposure \
        + start_time

    # log.info('reading observing schedule')
    # schedule = Table.read(args.schedule, format='ascii.ecsv')

    log.info('calculating field of regard')
    field_of_regard = get_field_of_regard(times)

    orbit_field_of_regard = np.logical_or.reduce(field_of_regard)
    continuous_viewing_zone = np.logical_and.reduce(field_of_regard)

    t = (times - times[0]).to(u.minute).value

    continuous_color, instantaneous_color, orbit_color = seaborn.color_palette(n_colors=3)

    fig = plt.figure(figsize=(8, 6))
    fig.suptitle('GUCI Field of Regard')
    gs_sky, gs_time = plt.GridSpec(2, 1, height_ratios=[2, 1])

    ax_time = fig.add_subplot(gs_time)
    ax_time.set_xlim(t[0], t[-1])
    ax_time.set_ylim(0, 100)
    ax_time.yaxis.set_major_formatter(FormatStrFormatter('%g%%'))
    ax_time.set_xlabel('Time (minutes)')
    ax_time.set_ylabel('Fraction of sky')
    twin = ax_time.twinx()
    twin.set_ylim(0, 4 * 180**2 / np.pi * 1e-4)
    twin.set_ylabel('Area ($10^4$ deg$^2$)')

    y = continuous_viewing_zone.sum() / skygrid.healpix.npix * 100
    ax_time.axhline(continuous_viewing_zone.sum() / skygrid.healpix.npix, color=continuous_color)
    ax_time.text(t[0], y, 'Continuous viewing', color=continuous_color, va='bottom')

    y = orbit_field_of_regard.sum() / skygrid.healpix.npix * 100
    ax_time.axhline(y, color=orbit_color)
    ax_time.text(t[0], y, 'Orbit-averaged', color=orbit_color, va='bottom')

    y = field_of_regard.sum(1) / skygrid.healpix.npix * 100
    ax_time.plot(t, y, color=instantaneous_color)

    ax_sky = fig.add_subplot(gs_sky, projection='astro mollweide')
    ax_sky.grid()

    old_artists = []

    def animate(i):
        for artist in old_artists:
            artist.remove()
        del old_artists[:]
        old_artists.extend(ax_sky.contourf_hpx(orbit_field_of_regard.astype(float), levels=[0.5, 2], colors=[orbit_color], nested=nest).collections)
        old_artists.extend(ax_sky.contourf_hpx(field_of_regard[i].astype(float), levels=[0.5, 2], colors=[instantaneous_color], nested=nest).collections)
        old_artists.extend(ax_sky.contourf_hpx(continuous_viewing_zone.astype(float), levels=[0.5, 2], colors=[continuous_color], nested=nest).collections)
        old_artists.append(ax_time.axvline(t[i], color='gray'))

    with tqdm(range(len(field_of_regard))) as frames:
        ani = FuncAnimation(fig, animate, frames=frames)
        ani.save(args.output.name)


if __name__ == '__main__':
    main()
