#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
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
