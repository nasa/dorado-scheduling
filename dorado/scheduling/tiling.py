#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Spacecraft survey model."""
from functools import partial

from astropy.coordinates import ICRS, SkyCoord, SkyOffsetFrame
from astropy_healpix import HEALPix
from astropy import units as u
import numpy as np
import healpy as hp

from .survey import SurveyModel


class TilingModel(SurveyModel):
    def __init__(self,
                 satfile='orbits.txt',
                 exposure_time=10 * u.minute,
                 time_steps_per_exposure=10,
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

        super().__init__(satfile, exposure_time, time_steps_per_exposure)

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
