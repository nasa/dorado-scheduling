#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft survey model."""

from astroplan import is_event_observable, Observer
from ligo.skymap.util import progress_map
from astropy import units as u
from astropy.coordinates import ICRS
from astropy_healpix import HEALPix
import numpy as np

from .constraints import OrbitNightConstraint


class SurveyModel():
    def __init__(self,
                 mission,
                 exposure_time=10 * u.minute,
                 time_steps_per_exposure=10,
                 number_of_orbits=1,
                 centers=None,
                 ):

        self.mission = mission
        self.healpix = HEALPix(nside=32, order='nested', frame=ICRS())
        """Base HEALpix resolution for all calculations."""

        if centers is None:
            self.centers = self.healpix.healpix_to_skycoord(
                np.arange(self.healpix.npix))
            """Centers of pointings."""
        else:
            self.centers = centers

        self.exposure_time = exposure_time
        self.time_steps_per_exposure = time_steps_per_exposure
        self.number_of_orbits = number_of_orbits
        self.time_step_duration = exposure_time / time_steps_per_exposure

        self.exposures_per_orbit = int(
            (self.mission.orbit.period /
             exposure_time).to_value(u.dimensionless_unscaled))
        self.time_steps = int(
            (self.number_of_orbits * self.mission.orbit.period /
             self.time_step_duration).to_value(u.dimensionless_unscaled))

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
            time, Observer(self.mission.orbit(time).earth_location))

    def _observable(self, time, location):
        return is_event_observable(
            self.mission.constraints,
            Observer(location),
            self.centers,
            time
        ).ravel()

    def get_field_of_regard(self, times, jobs=None):
        return np.asarray(list(progress_map(
            self._observable, times, self.mission.orbit(times).earth_location,
            jobs=jobs)))
