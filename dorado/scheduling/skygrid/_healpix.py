#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import ICRS
from astropy import units as u
from astropy_healpix import HEALPix
from ligo.skymap.bayestar.filter import ceil_pow_2
import numpy as np


def healpix(area):
    """Generate a grid in HEALPix coordinates.

    Parameters
    ----------
    area : :class:`astropy.units.Quantity`
        The average area per tile in any Astropy solid angle units:
        for example, :samp:`10 * astropy.units.deg**2` or
        :samp:`0.1 * astropy.units.steradian`.

    Returns
    -------
    coords : :class:`astropy.coordinates.SkyCoord`
        The coordinates of the tiles.

    """
    nside = np.sqrt(u.spat / (12 * area)).to_value(u.dimensionless_unscaled)
    nside = int(max(ceil_pow_2(nside), 1))
    hpx = HEALPix(nside, frame=ICRS())
    return hpx.healpix_to_skycoord(np.arange(hpx.npix))
