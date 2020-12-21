from astroplan.constraints import Constraint, _get_altaz
from astropy.constants import R_earth
import numpy as np

__all__ = ('EarthLimbConstraint',)


class EarthLimbConstraint(Constraint):
    """
    Constrain the angle from the Earth limb.

    Parameters
    ----------
    min : `~astropy.units.Quantity`
        Minimum angular separation from the Earth's limb.

    Notes
    -----
    This constraint assumes a spherical Earth, so it is only accurate to about
    a degree for observers in very low Earth orbit (height of 100 km).
    """

    def __init__(self, min):
        self.min = min

    def compute_constraint(self, times, observer, targets):
        cached_altaz = _get_altaz(times, observer, targets)
        alt = cached_altaz['altaz'].alt
        h_r = np.maximum(0, observer.location.height / R_earth)
        limb_alt = np.arccos(1 / (1 + h_r))
        print(observer.location.height, h_r, limb_alt)
        return alt >= self.min - limb_alt