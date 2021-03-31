#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft survey model."""
from importlib import resources

from astroplan import is_event_observable, Observer
from ligo.skymap.util import progress_map
from astropy.coordinates import SkyCoord, TEME
from astropy import units as u
import numpy as np
from sgp4.api import Satrec, SGP4_ERRORS

from . import data
from .constraints import OrbitNightConstraint
from .constraints import visibility_constraints


class SurveyModel():
    def __init__(self,
                 satfile='orbits.txt',
                 exposure_time=10 * u.minute,
                 time_steps_per_exposure=10,
                 ):

        self.satfile = satfile
        self.exposure_time = exposure_time
        self.time_steps_per_exposure = time_steps_per_exposure

        # Load two-line element for satellite.
        # This is for Aqua, an Earth observing satellite in a low-Earth
        # sun-synchronous orbit that happens to be similar to what might
        # be appropriate for Dorado.
        with resources.open_text(data, satfile) as f:
            _, line1, line2 = f.readlines()
            self.satellite = Satrec.twoline2rv(line1, line2)

        self.orbital_period = 2 * np.pi / self.satellite.no * u.minute
        self.time_step_duration = exposure_time / time_steps_per_exposure

        self.exposures_per_orbit = int(
            (self.orbital_period /
             exposure_time).to_value(u.dimensionless_unscaled))
        self.time_steps = int(
            (self.orbital_period /
             self.time_step_duration).to_value(u.dimensionless_unscaled))

    def get_posvel(self, time):
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
        e, xyz, vxyz = self.satellite.sgp4_array(time.jd1, time.jd2)
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

    def is_night(self, time):
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
            time, Observer(self.get_posvel(time).earth_location))

    def _observable(self, time, location):
        return is_event_observable(
            visibility_constraints,
            Observer(location),
            self.centers,
            time
        ).ravel()

    def get_field_of_regard(self, times, jobs=None):
        return np.asarray(list(progress_map(
            self._observable, times, self.get_posvel(times).earth_location,
            jobs=jobs)))
