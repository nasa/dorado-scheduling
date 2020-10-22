"""Grid of pointings on the sky."""
from astropy_healpix import HEALPix
from astropy.coordinates import ICRS, SkyCoord
from astropy import units as u
from astropy.wcs import WCS
import healpy as hp
from ligo.skymap.util import progress_map
import numpy as np

__all__ = ('healpix', 'centers', 'rolls', 'field_of_view',
           'get_footprint_grid')

healpix = HEALPix(nside=32, order='nested', frame=ICRS())
"""Base HEALpix resolution for all calculations."""

centers = healpix.healpix_to_skycoord(np.arange(healpix.npix))
"""Centers of pointings."""

rolls = np.linspace(0, 45, 9, endpoint=False) * u.deg
"""Roll angle grid. Only goes up to 45° because field of view is a square."""

field_of_view = 7.1 * u.deg
"""Width of the (square) field of view."""


def get_footprint_polygon(center, rotate=None):
    """Get the footprint of the field of view for a given orientation.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        The center of the field of view.
    rotate : astropy.units.Quantity
        The position angle (optional, default 0 degrees).

    Returns
    -------
    astropy.coordinates.SkyCoord
        A sky coordinate array giving the four verticies of the footprint.

    """
    center = SkyCoord(center).icrs
    radius = 0.5 * field_of_view.to_value(u.deg)
    header = {
        'NAXIS': 2,
        'NAXIS1': 512,
        'NAXIS2': 512,
        'CRPIX1': 256.5,
        'CRPIX2': 256.5,
        'CRVAL1': center.ra.deg,
        'CRVAL2': center.dec.deg,
        'CDELT1': -radius / 256,
        'CDELT2': radius / 256,
        'CTYPE1': 'RA---TAN',
        'CTYPE2': 'DEC--TAN',
        'RADESYS': 'ICRS'}
    if rotate is not None:
        header['LONPOLE'] = u.Quantity(rotate).to_value(u.deg)
    return SkyCoord(WCS(header).calc_footprint(), unit=u.deg)


def get_footprint_healpix(center, rotate=None):
    """Get the footprint of the field of view for a given orientation.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        The center of the field of view.
    rotate : astropy.units.Quantity
        The position angle (optional, default 0 degrees).

    Returns
    -------
    numpy.ndarray
        Array of HEALPix coordinates within the field of view.

    """
    polygon = get_footprint_polygon(center, rotate)
    xyz = polygon.cartesian.xyz.value.T
    return hp.query_polygon(healpix.nside, xyz,
                            nest=(healpix.order == 'nested'))


def get_footprint_grid():
    """Calculate the HEALPix footprints of all pointings on the grid.

    Returns
    -------
    np.ndarray
        An array of HEALPix footprints of size ``(healpix.npix, rolls.size)``
        where each element is a list of HEALPix indices within the grid.

    """
    center_grid, roll_grid = np.meshgrid(centers, rolls, indexing='ij')
    footprints = progress_map(get_footprint_healpix,
                              center_grid.ravel(),
                              roll_grid.ravel(),
                              jobs=None)

    # Manually reshape, because array is ragged
    return ((footprint for footprint, j in zip(footprints, range(len(rolls))))
            for i in range(len(centers)))
