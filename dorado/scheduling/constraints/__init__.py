#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Astroplan visibility constraints."""
from functools import partial

import astroplan
from astropy import units as u
from ligo.skymap.util import progress_map
import numpy as np

from .earth_limb import EarthLimbConstraint
from .orbit_night import OrbitNightConstraint
from .radiation import TrappedParticleFluxConstraint

__all__ = ('get_field_of_regard',
           'visibility_constraints',
           'EarthLimbConstraint',
           'OrbitNightConstraint',
           'OutsideSouthAtlanticAnomalyConstraint',
           'TrappedParticleFluxConstraint')

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
    astroplan.MoonSeparationConstraint(23 * u.deg),
    # 10° from Galactic plane
    astroplan.GalacticLatitudeConstraint(10 * u.deg)
]
"""Example observability constraints from Swift.

See also
--------
`Swift Technical Handbook
<https://swift.gsfc.nasa.gov/proposals/tech_appd/swiftta_v14/node24.html>`_
"""


def _observable(targets, time, location):
    return astroplan.is_event_observable(
        visibility_constraints,
        astroplan.Observer(location),
        targets,
        time
    ).ravel()


def get_field_of_regard(orbit, targets, times, jobs=None):
    """Calculate the observability of a grid of targets at a grid of times.

    Parameters
    ----------
    orbit : :class:`dorado.scheduling.Orbit`
        The orbit of the satellite.
    targets : :class:`astropy.coordinates.SkyCoord`
        An array of coordinates of size N.
    times : :class:`astropy.coordinates.SkyCoord`
        An array of times of size M.
    jobs : int
        The number of threads to use, or None to use all available cores.

    Returns
    -------
    regard : :class:`np.ndarray`
        A boolean array of size M×N, which is true if a given target is
        observable at a given time.
    """
    return np.asarray(list(progress_map(
        partial(_observable, targets),
        times, orbit(times).earth_location, jobs=jobs)))
