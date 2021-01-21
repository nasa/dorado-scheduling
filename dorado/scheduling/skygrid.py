#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Grid of pointings on the sky."""
from functools import partial

from astropy_healpix import HEALPix
from astropy.coordinates import ICRS, SkyCoord, SkyOffsetFrame
from astropy import units as u
import healpy as hp
import numpy as np

__all__ = ('healpix', 'centers', 'rolls', 'field_of_view',
           'get_footprint_polygon', 'get_footprint_healpix',
           'get_footprint_grid')

healpix = HEALPix(nside=32, order='nested', frame=ICRS())
"""Base HEALpix resolution for all calculations."""

centers = healpix.healpix_to_skycoord(np.arange(healpix.npix))
"""Centers of pointings."""

rolls = np.linspace(0, 90, 9, endpoint=False) * u.deg
"""Roll angle grid."""

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
    frame = SkyOffsetFrame(origin=center, rotation=rotate)
    lon = np.asarray([0.5, 0.5, -0.5, -0.5]) * field_of_view
    lat = np.asarray([0.5, -0.5, -0.5, 0.5]) * field_of_view
    return SkyCoord(
        np.tile(lon[(None,) * frame.ndim], (*frame.shape, 1)),
        np.tile(lat[(None,) * frame.ndim], (*frame.shape, 1)),
        frame=frame[..., None]
    ).icrs


def get_footprint_healpix(center, rotate=None):
    """Get the HEALPix footprint of the field of view for a given orientation.

    Parameters
    ----------
    center : astropy.coordinates.SkyCoord
        The center of the field of view.
    rotate : astropy.units.Quantity
        The position angle (optional, default 0 degrees).

    Returns
    -------
    np.ndarray
        An array of HEALPix indices contained within the footprint.

    """
    xyz = get_footprint_polygon(center, rotate=rotate).cartesian.xyz.value
    return hp.query_polygon(healpix.nside, xyz.T,
                            nest=(healpix.order == 'nested'))


def get_footprint_grid():
    """Calculate the HEALPix footprints of all pointings on the grid.

    Returns
    -------
    generator
        A generator that yields the indices for each pointing center and for
        each roll.

    """
    xyz = np.moveaxis(
        get_footprint_polygon(
            centers[:, None], rolls[None, :]).cartesian.xyz.value, 0, -1)
    query_polygon = partial(
        hp.query_polygon, healpix.nside, nest=(healpix.order == 'nested'))
    return ((query_polygon(_) for _ in __) for __ in xyz)
