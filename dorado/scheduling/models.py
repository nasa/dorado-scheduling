#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft survey model."""
from functools import partial
from importlib import resources

import astroplan
from astroplan import is_event_observable, Observer
from ligo.skymap.util import progress_map
from astropy.coordinates import SkyCoord, TEME
from astropy import units as u
from astropy.coordinates import ICRS, SkyOffsetFrame
from astropy_healpix import HEALPix
import numpy as np
import healpy as hp
from sgp4.api import Satrec, SGP4_ERRORS

from . import data
from .constraints import OrbitNightConstraint
from .constraints.earth_limb import EarthLimbConstraint
from .constraints.radiation import TrappedParticleFluxConstraint

visibility_constraints = [
    # SAA constraint, modeled after Fermi:
    # flux of particles with energies ≥ 20 MeV is ≤ 1 cm^-2 s^-1
    TrappedParticleFluxConstraint(flux=1*u.cm**-2*u.s**-1, energy=20*u.MeV,
                                  particle='p', solar='max'),
    # 28° from the Earth's limb
    EarthLimbConstraint(28 * u.deg),
    # 46° from the Sun
    astroplan.SunSeparationConstraint(46 * u.deg),
    # 23° from the Moon
    astroplan.MoonSeparationConstraint(23 * u.deg)
    # 10° from Galactic plane
    # astroplan.GalacticLatitudeConstraint(10 * u.deg)
]


class SurveyModel():
    def __init__(self,
                 satfile='orbits.txt',
                 exposure_time=10 * u.minute,
                 time_steps_per_exposure=10,
                 number_of_orbits=1,
                 ):

        self.satfile = satfile
        self.exposure_time = exposure_time
        self.time_steps_per_exposure = time_steps_per_exposure
        self.number_of_orbits = number_of_orbits

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
            (self.number_of_orbits * self.orbital_period /
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


class TilingModel(SurveyModel):
    def __init__(self,
                 satfile='orbits.txt',
                 exposure_time=10 * u.minute,
                 time_steps_per_exposure=10,
                 number_of_orbits=1,
                 field_of_view=7.1 * u.deg,
                 centers=None
                 ):

        self.healpix = HEALPix(nside=32, order='nested', frame=ICRS())
        """Base HEALpix resolution for all calculations."""

        if centers is None:
            self.centers = self.healpix.healpix_to_skycoord(
                np.arange(self.healpix.npix))
            """Centers of pointings."""
        else:
            self.centers = centers

        self.rolls = np.linspace(0, 90, 9, endpoint=False) * u.deg
        """Roll angle grid."""

        self.field_of_view = field_of_view
        """Width of the (square) field of view."""

        super().__init__(satfile, exposure_time, time_steps_per_exposure,
                         number_of_orbits)

    def get_footprint_polygon(self, center, rotate=None):
        """Get the footprint of the field of view for a given orientation.

        Parameters
        ----------
        center : :class:`astropy.coordinates.SkyCoord`
            The center of the field of view.
        rotate : :class:`astropy.units.Quantity`
            The position angle (optional, default 0 degrees).

        Returns
        -------
        :class:`astropy.coordinates.SkyCoord`
            A sky coordinate array giving the four verticies of the footprint.

        """
        frame = SkyOffsetFrame(origin=center, rotation=rotate)
        lon = np.asarray([0.5, 0.5, -0.5, -0.5]) * self.field_of_view
        lat = np.asarray([0.5, -0.5, -0.5, 0.5]) * self.field_of_view
        return SkyCoord(
            np.tile(lon[(None,) * frame.ndim], (*frame.shape, 1)),
            np.tile(lat[(None,) * frame.ndim], (*frame.shape, 1)),
            frame=frame[..., None]
        ).icrs

    def get_footprint_healpix(self, center, rotate=None):
        """Get the HEALPix footprint of the field of view for a given orientation.

        Parameters
        ----------
        center : :class:`astropy.coordinates.SkyCoord`
            The center of the field of view.
        rotate : class:`astropy.units.Quantity`
            The position angle (optional, default 0 degrees).

        Returns
        -------
        class:`np.ndarray`
            An array of HEALPix indices contained within the footprint.

        """
        xyz = self.get_footprint_polygon(center,
                                         rotate=rotate).cartesian.xyz.value
        return hp.query_polygon(self.healpix.nside, xyz.T,
                                nest=(self.healpix.order == 'nested'))

    def get_footprint_grid(self):
        """Calculate the HEALPix footprints of all pointings on the grid.

        Returns
        -------
        generator
            A generator that yields the indices for each pointing center
            and for each roll.

        """
        xyz = np.moveaxis(
            self.get_footprint_polygon(
                self.centers[:, None],
                self.rolls[None, :]).cartesian.xyz.value, 0, -1)
        query_polygon = partial(
            hp.query_polygon, self.healpix.nside,
            nest=(self.healpix.order == 'nested'))
        return ((query_polygon(_) for _ in __) for __ in xyz)
