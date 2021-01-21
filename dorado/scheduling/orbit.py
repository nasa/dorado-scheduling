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
from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import numpy as np
import skyfield.api

from . import data
from .constraints import OrbitNightConstraint

__all__ = ('get_position', 'orbital_period', 'exposure_time',
           'exposures_per_orbit')


# Load two-line element for satellite.
# This is for Aqua, an Earth observing satellite in a low-Earth sun-synchronous
# orbit that happens to be similar to what might be appropriate for Dorado.
with resources.path(data, 'orbits.txt') as path:
    satellite = skyfield.api.load.tle(str(path))['AQUA']

timescale = skyfield.api.load.timescale()

orbital_period = 2 * np.pi / satellite.model.no * u.minute
exposure_time = 10 * u.minute
time_steps_per_exposure = 10
time_step_duration = exposure_time / time_steps_per_exposure
exposures_per_orbit = int(
    (orbital_period / exposure_time).to_value(u.dimensionless_unscaled))
time_steps = int(
    (orbital_period / time_step_duration).to_value(u.dimensionless_unscaled))


def get_position(time):
    """Get the position of the satellite.

    Parameters
    ----------
    time : astropy.time.Time, skyfield.timelib.Time
        The time of the observation.

    Returns
    -------
    earth_location : astropy.coordinates.EarthLocation
        The geocentric position of the satellite.
    """
    if isinstance(time, Time):
        time = timescale.from_astropy(time)
    position = satellite.at(time).position
    return EarthLocation.from_geocentric(*position.to(u.meter))


def is_night(time):
    """Determine if the spacecraft is in orbit night.

    Parameters
    ----------
    time : astropy.time.Time, skyfield.timelib.Time
        The time of the observation.

    Returns
    -------
    bool, np.ndarray
        True when the spacecraft is in orbit night, False otherwise.
    """
    return OrbitNightConstraint().compute_constraint(
        time, Observer(get_position(time)))
