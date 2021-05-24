#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import Angle
from astropy.coordinates.matrix_utilities import (
    matrix_product, rotation_matrix)
from astropy import units as u
import numpy as np


def slew_time(x, v, a):
    """Calculate the time to execute an optimal slew of a given distance.

    The optimal slew consists of an acceleration phase at the maximum
    acceleration, possibly a coasting phase at the maximum angular velocity,
    and a deceleration phase at the maximum acceleration.

    Parameters
    ----------
    x : float, numpy.ndarray
        Distance.
    v : float, numpy.ndarray
        Maximum velocity.
    a : float, numpy.ndarray
        Maximum acceleration.

    Returns
    -------
    t : float, numpy.ndarray
        The minimum time to slew through a distance ``x`` given maximum
        velocity ``v`` and maximum acceleration ``a``.

    """
    xc = np.square(v) / a
    return np.where(x <= xc, np.sqrt(4 * x / a), (x + xc) / v)


def matrix_trace(matrix):
    return np.trace(matrix, axis1=-2, axis2=-1)


def slew_separation(center1, center2, roll1=0 * u.rad, roll2=0 * u.rad):
    """
    Determine the smallest angle to slew between two attitudes.

    Parameters
    ----------
    center1 : :class:`astropy.coordinates.SkyCoord`
    center2 : :class:`astropy.coordinates.SkyCoord`
    roll1 : :class:`astropy.units.Angle`
    roll2 : :class:`astropy.units.Angle`

    Returns
    -------
    angle : :class:`astropy.units.Angle`

    Examples
    --------
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as u
    >>> c1 = SkyCoord(0 * u.deg, 20 * u.deg)
    >>> c2 = SkyCoord(0 * u.deg, 40 * u.deg)
    >>> roll1 = 20 * u.deg
    >>> roll2 = 40 * u.deg
    >>> slew_separation(c1, c2)
    <Angle 20. deg>
    >>> slew_separation(c1, c1, roll1, roll2)
    <Angle 20. deg>
    >>> slew_separation(c1, c2, roll1, roll2)
    <Angle 28.21208852 deg>

    """
    assert center1.is_equivalent_frame(center2)
    center1 = center1.spherical
    center2 = center2.spherical
    mat = matrix_product(rotation_matrix(roll2 - roll1, 'x'),
                         rotation_matrix(-center1.lat, 'y'),
                         rotation_matrix(center1.lon - center2.lon, 'z'),
                         rotation_matrix(center2.lat, 'y'))
    return Angle(np.arccos(0.5 * (matrix_trace(mat) - 1)), unit=u.rad)
