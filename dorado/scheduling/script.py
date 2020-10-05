"""Command line interface."""
import logging

from astropy_healpix import nside_to_level
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from ligo.skymap.tool import ArgumentParser, FileType
import mip
import numpy as np

from .orbit import satellite
from .regard import get_field_of_regard
from .skygrid import get_footprint_grid, healpix, rolls

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    return p


def main(args=None):
    args = parser().parse_args(args)

    log.info('reading sky map')

    # Read multi-order sky map and rasterize to working resolution
    skymap = read_sky_map(args.skymap, moc=True)
    skymap['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, nside_to_level(healpix.nside))['PROB']
    if healpix.order == 'ring':
        prob = prob[healpix.ring_to_nested(np.arange(len(prob)))]

    log.info('calculating field of regard')

    start_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    # start_time = skymap.meta['obstime']
    orbital_period = 2 * np.pi / satellite.model.no * u.minute
    exposure_time = 10 * u.minute
    max_exposures_per_orbit = int(np.ceil((
        orbital_period / exposure_time).to_value(u.dimensionless_unscaled)))
    time_steps_per_exposure = 10  # simulation time steps per exposure
    time_steps = max_exposures_per_orbit * time_steps_per_exposure
    times = start_time + orbital_period * np.linspace(0, 1, time_steps)

    field_of_regard = get_field_of_regard(times)

    log.info('building footprint grid')

    grid_inverse = [[] for _ in range(healpix.npix)]
    for i, grid_i in enumerate(get_footprint_grid()):
        for j, grid_ij in enumerate(grid_i):
            for k in grid_ij:
                grid_inverse[k].append((i, j))

    log.info('building MIP model')

    m = mip.Model()

    field_observed = m.add_var_tensor(
        (healpix.npix, len(rolls)), 'field_observed', var_type=mip.BINARY)
    pixel_observed = m.add_var_tensor(
        (healpix.npix,), 'pixel_observed', var_type=mip.BINARY)

    for k, ij in enumerate(grid_inverse):
        m += mip.xsum(field_observed[i][j] for i, j in ij) >= pixel_observed[k]
    m.objective = mip.maximize(mip.xsum(prob * pixel_observed))


if __name__ == '__main__':
    main()
