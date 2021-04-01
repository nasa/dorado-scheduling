#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def sinusoidal(area):
    """Generate a uniform grid on a sinusoidal equal area projection.

    This is similar to what was used for GRANDMA follow-up in LIGO/Virgo
    Observing Run 3 (O3). See :doi:`10.3847/2041-8213/ab3399`.

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
    <https://en.wikipedia.org/wiki/Sinusoidal_projection>

    """
    # Diameter of the field of view
    diameter = 2 * np.sqrt(area.to_value(u.sr) / np.pi)

    # Declinations of equal-declination strips
    n_decs = int(np.ceil(np.pi / diameter)) + 1
    decs = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_decs)

    # Number of RA steps in each equal-declination strip
    n_ras = np.ceil(2 * np.pi * np.cos(decs) / diameter).astype(int)
    n_ras = np.maximum(1, n_ras)

    ras = np.concatenate([np.linspace(0, 2 * np.pi, n, endpoint=False)
                          for n in n_ras])
    decs = np.concatenate([np.repeat(dec, n) for n, dec in zip(n_ras, decs)])
    return SkyCoord(ras * u.rad, decs * u.rad)
