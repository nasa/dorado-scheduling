#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft orbit."""
from importlib import resources

from astroplan import Observer
from astropy.coordinates import SkyCoord, TEME
from astropy import units as u
import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS

from . import data
from .constraints import OrbitNightConstraint

__all__ = ('get_posvel', 'orbital_period', 'exposure_time',
           'exposures_per_orbit', 'is_night')

# Load two-line element for satellite.
# This is for Aqua, an Earth observing satellite in a low-Earth sun-synchronous
# orbit that happens to be similar to what might be appropriate for Dorado.
with resources.open_text(data, 'orbits.txt') as f:
    _, line1, line2 = f.readlines()
    satellite = Satrec.twoline2rv(line1, line2)

orbital_period = 2 * np.pi / satellite.no * u.minute
exposure_time = 10 * u.minute
time_steps_per_exposure = 10
time_step_duration = exposure_time / time_steps_per_exposure
exposures_per_orbit = int(
    (orbital_period / exposure_time).to_value(u.dimensionless_unscaled))
time_steps = int(
    (orbital_period / time_step_duration).to_value(u.dimensionless_unscaled))


def get_posvel(time):
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
    This is based on
    https://docs.astropy.org/en/stable/coordinates/satellites.html.
    """
    shape = time.shape
    time = time.ravel()

    time = time.utc
    e, xyz, vxyz = satellite.sgp4_array(time.jd1, time.jd2)
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


def is_night(time):
    """Determine if the spacecraft is in orbit night.

    Parameters
    ----------
    time : :class:`astropy.time.Time`
        The time of the observation.

    Returns
    -------
    bool, :class:`np.ndarray`
        True when the spacecraft is in orbit night, False otherwise.
    """
    return OrbitNightConstraint().compute_constraint(
        time, Observer(get_posvel(time).earth_location))
