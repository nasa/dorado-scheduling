#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft orbit."""
from astropy.coordinates import SkyCoord, TEME
from astropy import units as u
from astropy.utils.data import get_readable_fileobj
import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS

__all__ = ('Orbit',)


class Orbit:
    """An Earth satellite whose orbit is specified by its TLE.

    Parameters
    ----------
    tle : str, file
        The filename or file-like object containing the two-line element (TLE).

    Examples
    --------

    Load an example TLE from a file:

    >>> from importlib import resources
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from dorado.scheduling import Orbit
    >>> from astropy.utils.data import get_pkg_data_filename
    >>> with resources.path('dorado.scheduling.data', 'orbits.txt') as path:
    ...     orbit = Orbit(path)

    Get the orbital period:

    >>> orbit.period
    <Quantity 98.82566607 min>

    Evaluate the position and velocity of the satellite at one specific time:

    >>> time = Time('2021-04-16 15:27')
    >>> orbit(time)
    <SkyCoord (ITRS: obstime=2021-04-16 15:27:00.000): (x, y, z) in km
        (4976.75920356, -3275.77173105, 3822.71352292)
     (v_x, v_y, v_z) in km / s
        (-4.28131812, 0.77514995, 6.2200722)>

    Or evaluate at an array of times:

    >>> times = time + np.linspace(0 * u.min, 2 * u.min, 3)
    >>> orbit(times)
    <SkyCoord (ITRS: obstime=['2021-04-16 15:27:00.000' '2021-04-16 15:28:00.000'
     '2021-04-16 15:29:00.000']): (x, y, z) in km
        [(4976.75920356, -3275.77173105, 3822.71352292),
         (4710.26989911, -3221.55488875, 4187.9218208 ),
         (4425.37931006, -3151.96969293, 4536.16324051)]
     (v_x, v_y, v_z) in km / s
        [(-4.28131812, 0.77514995, 6.2200722 ),
         (-4.59829403, 1.0319373 , 5.94941852),
         (-4.89448943, 1.28722498, 5.65470357)]>

    """  # noqa: E501

    def __init__(self, tle):
        with get_readable_fileobj(tle) as f:
            *_, line1, line2 = f.readlines()
        self._tle = Satrec.twoline2rv(line1, line2)

    @property
    def period(self):
        """The orbital period at the epoch of the TLE."""
        return 2 * np.pi / self._tle.no * u.minute

    def __call__(self, time):
        """Get the position and velocity of the satellite.

        Parameters
        ----------
        time : :class:`astropy.time.Time`
            The time of the observation.

        Returns
        -------
        coord : :class:`astropy.coordinates.SkyCoord`
            The coordinates of the satellite in the ITRS frame.

        Notes
        -----
        The orbit propagation is based on the example code at
        https://docs.astropy.org/en/stable/coordinates/satellites.html.

        """
        shape = time.shape
        time = time.ravel()

        time = time.utc
        e, xyz, vxyz = self._tle.sgp4_array(time.jd1, time.jd2)
        x, y, z = xyz.T
        vx, vy, vz = vxyz.T

        # If any errors occurred, only raise for the first error
        e = e[e != 0]
        if e.size > 0:
            raise RuntimeError(SGP4_ERRORS[e[0]])

        coord = SkyCoord(x=x*u.km, v_x=vx*u.km/u.s,
                         y=y*u.km, v_y=vy*u.km/u.s,
                         z=z*u.km, v_z=vz*u.km/u.s,
                         frame=TEME(obstime=time)).itrs
        if shape:
            coord = coord.reshape(shape)
        else:
            coord = coord[0]
        return coord
