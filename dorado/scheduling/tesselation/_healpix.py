#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
import math

from astropy.coordinates import ICRS
from astropy import units as u
from astropy_healpix import HEALPix
import numpy as np


def ceil_pow_2(n):
    """Return the least integer power of 2 that is greater than or equal to n.

    Examples
    --------
    >>> ceil_pow_2(128.0)
    128.0
    >>> ceil_pow_2(0.125)
    0.125
    >>> ceil_pow_2(129.0)
    256.0
    >>> ceil_pow_2(0.126)
    0.25
    >>> ceil_pow_2(1.0)
    1.0

    """
    # frexp splits floats into mantissa and exponent, ldexp does the opposite.
    # For positive numbers, mantissa is in [0.5, 1.).
    mantissa, exponent = math.frexp(n)
    return math.ldexp(
        1 if mantissa >= 0 else float('nan'),
        exponent - 1 if mantissa == 0.5 else exponent
    )


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
