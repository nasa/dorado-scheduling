#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.coordinates import SkyCoord, UnitSphericalRepresentation
from astropy import units as u
from cdshealpix.nested import polygon_search
import numpy as np

__all__ = ('FOV',)


class FOV:
    """The field of view of an instrument.

    Parameters
    ----------
    representation : :class:`astropy.coordinates.BaseRepresentation`
        The coordinates of the vertices of the field of view in any Astropy
        representation.

    """

    def __init__(self, representation):
        self._representation = representation

    @classmethod
    def from_rectangle(cls, width, height=None):
        """Create a rectangular field of view.

        Parameters
        ----------
        width : :class:`astropy.units.Quantity`
            The width of the rectangle.
        height : :class:`astropy.units.Quantity`
            The height of the rectangle. If omitted, then the field of view is
            a square.

        Returns
        -------
        :class:`FOV`

        """
        if height is None:
            height = width
        lon = np.asarray([0.5, 0.5, -0.5, -0.5]) * width
        lat = np.asarray([0.5, -0.5, -0.5, 0.5]) * height
        return cls(UnitSphericalRepresentation(lon, lat))

    def footprint(self, center=SkyCoord(0*u.deg, 0*u.deg), roll=0*u.deg):
        """Get the footprint of the FOV at a given orientation.

        Parameters
        ----------
        center : :class:`astropy.coordinates.SkyCoord`
            The center of the field of view. If omitted, the default is
            RA=0°, Dec=0°.
        roll : :class:`astropy.coordinates.SkyCoord`
            The roll of the field of view. If omitted, the default is 0°.

        Returns
        -------
        vertices : `astropy.coordinates.SkyCoord`
            The coordinates of the vertices of the field of view.

        Examples
        --------
        First, some imports:

        >>> from astropy.coordinates import ICRS, SkyCoord
        >>> from astropy import units as u
        >>> from astropy_healpix import HEALPix

        Now, get the footprint for the default pointing:

        >>> fov = FOV.from_rectangle(50 * u.deg)
        >>> fov.footprint().icrs
        <SkyCoord (ICRS): (ra, dec) in deg
            [( 25.,  25.), ( 25., -25.), (335., -25.), (335.,  25.)]>

        Get the footprint for a spcific pointing:

        >>> fov.footprint(SkyCoord(0*u.deg, 20*u.deg), 15*u.deg).icrs
        <SkyCoord (ICRS): (ra, dec) in deg
            [( 35.73851095,  34.84634609), ( 15.41057822, -11.29269295),
             (331.35544934,  -0.54495686), (336.46564791,  49.26075815)]>

        Get a footprint as HEALPix coordinates.

        >>> hpx = HEALPix(nside=4, frame=ICRS())
        >>> fov.footprint_healpix(hpx, SkyCoord(0*u.deg, 20*u.deg), 15*u.deg)
        array([ 57,  41,  25,  24,  70,  55,  39,  38,  23, 120, 104, 105,  88,
                73, 119, 103, 102,  87,  72,  56,  71,  40])

        Get the footprint for a grid of pointings:

        >>> ra = np.arange(0, 360, 45) * u.deg
        >>> dec = np.arange(-90, 91, 45) * u.deg
        >>> roll = np.arange(0, 90, 15) * u.deg
        >>> center = SkyCoord(ra[:, np.newaxis], dec[np.newaxis, :])
        >>> footprints = fov.footprint(center[..., np.newaxis], roll)
        >>> footprints.shape
        (8, 5, 6, 4)

        """  # noqa: E501
        frame = center.skyoffset_frame(roll)[..., np.newaxis]
        representation = self._representation

        # FIXME: Astropy does not automatically broadcast frame attributes.
        # Remove once https://github.com/astropy/astropy/issues/8812 is fixed.
        shape = np.broadcast_shapes(representation.shape, frame.shape)
        frame = np.broadcast_to(frame, shape, subok=True)
        representation = np.broadcast_to(self._representation, shape,
                                         subok=True)

        # FIXME: Astropy does not support realizing frames with representations
        # that are broadcast views. Remove once
        # https://github.com/astropy/astropy/issues/11572 is fixed.
        representation = representation.copy()

        return SkyCoord(frame.realize_frame(representation))

    @staticmethod
    def _polygon_search_internal(healpix, vertices):
        ipix, _, _ = polygon_search(
            vertices.lon, vertices.lat, depth=healpix.level, flat=True)
        ipix = ipix.astype(np.intp)
        if healpix.order == 'ring':
            ipix = healpix.nested_to_ring(ipix)
        return ipix

    def footprint_healpix(self, healpix, *args, **kwargs):
        """Get the HEALPix footprint at a given orientation.

        Parameters
        ----------
        center : :class:`astropy.coordinates.SkyCoord`
            The center of the field of view.
        rotate : class:`astropy.units.Quantity`
            The position angle (optional, default 0 degrees).

        Returns
        -------
        :class:`np.ndarray`
            An array of HEALPix indices contained within the footprint.

        """
        vertices = self.footprint(*args, **kwargs).transform_to(
            healpix.frame).represent_as(UnitSphericalRepresentation)
        return self._polygon_search_internal(healpix, vertices)

    def footprint_healpix_grid(self, healpix, center, roll):
        """Calculate the HEALPix footprints of all pointings on the grid.

        Returns
        -------
        generator
            A generator that yields the indices for each pointing center and
            for each roll.

        """
        vertices = self.footprint(center[:, None], roll[None, :]).transform_to(
            healpix.frame).represent_as(UnitSphericalRepresentation)
        return ((self._polygon_search_internal(healpix, _) for _ in __)
                for __ in vertices)
