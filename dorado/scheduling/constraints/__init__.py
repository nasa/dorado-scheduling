"""Astroplan visibility constraints."""
import astroplan
from astropy import units as u

from .saa import OutsideSouthAtlanticAnomalyConstraint
from ..skygrid import healpix

__all__ = ('visbility_constraints',)

# Example observability constraints from the
# Swift Technical Handbook
# <https://swift.gsfc.nasa.gov/proposals/tech_appd/swiftta_v14/node24.html>.
visbility_constraints = [
    # SAA constraint
    OutsideSouthAtlanticAnomalyConstraint(healpix.nside),
    # 28° from the Earth's limb (95° from the center of the Earth),
    # so 5° below "horizon"
    astroplan.AltitudeConstraint(5 * u.deg),
    # 46° from the Sun
    astroplan.SunSeparationConstraint(46 * u.deg),
    # 23° from the Moon
    astroplan.MoonSeparationConstraint(23 * u.deg),
    # 10° from Galactic plane
    astroplan.GalacticPlaneSeparationConstraint(10 * u.deg)
]
