#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from .earth_limb import EarthLimbConstraint
from .orbit_night import OrbitNightConstraint

import numpy as np

__all__ = ('BrightEarthLimbConstraint',)


class BrightEarthLimbConstraint(EarthLimbConstraint):
    """
    Constrain the angle from the Sun-illuminated portion of the Earth limb.

    Parameters
    ----------
    min : :class:`astropy.units.Quantity`
        Minimum angular separation from the Earth's limb.

    Notes
    -----
    This makes the extremely conservative simplifying assumption that the
    entire Earth is illuminated except during orbit night.

    """

    def __init__(self, *args):
        super().__init__(*args)
        self._orbit_night_constraint = OrbitNightConstraint()

    def compute_constraint(self, times, observer, targets):
        is_night = self._orbit_night_constraint.compute_constraint(
            times, observer, targets)
        above_limb = super().compute_constraint(
            times, observer, targets)
        return np.where(is_night, True, above_limb)
