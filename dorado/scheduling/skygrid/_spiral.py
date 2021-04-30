#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np

GOLDEN_ANGLE = np.pi * (3 - np.sqrt(5)) * u.rad


def golden_angle_spiral(area):
    """Generate a tile grid from a spiral employing the golden angle.

    This is a spiral-based spherical packing scheme that was used by GRANDMA
    during LIGO/Virgo O3 (see :doi:`10.1093/mnras/staa1846`).

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

    See also
    --------
    <https://en.wikipedia.org/wiki/Golden_angle>

    """
    n = int(np.ceil(1 / area.to_value(u.spat)))
    ra = GOLDEN_ANGLE * np.arange(n)
    dec = np.arcsin(np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)) * u.rad
    return SkyCoord(ra, dec)
