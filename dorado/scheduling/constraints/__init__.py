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
from ligo.skymap.util import progress_map
import numpy as np

from .earth_limb import EarthLimbConstraint
from .orbit_night import OrbitNightConstraint
from .radiation import TrappedParticleFluxConstraint

__all__ = ('get_field_of_regard',
           'EarthLimbConstraint',
           'OrbitNightConstraint',
           'OutsideSouthAtlanticAnomalyConstraint',
           'TrappedParticleFluxConstraint')


def _observable(constraints, targets, time, location):
    return astroplan.is_event_observable(
        constraints,
        astroplan.Observer(location),
        targets,
        time
    ).ravel()


def get_field_of_regard(orbit, constraints, targets, times, jobs=None):
    """Calculate the observability of a grid of targets at a grid of times.

    Parameters
    ----------
    orbit : :class:`dorado.scheduling.Orbit`
        The orbit of the satellite.
    constraints : list
        List of Astroplan constraints.
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
        partial(_observable, constraints, targets),
        times, orbit(times).earth_location, jobs=jobs)))
