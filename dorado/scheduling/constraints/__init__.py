"""Astroplan visibility constraints."""
import astroplan
from astropy import units as u

from .earth_limb import EarthLimbConstraint
from .orbit_night import OrbitNightConstraint
from .saa import OutsideSouthAtlanticAnomalyConstraint
from ..skygrid import healpix

__all__ = ('visibility_constraints',
           'EarthLimbConstraint',
           'OrbitNightConstraint',
           'OutsideSouthAtlanticAnomalyConstraint')

# Example observability constraints from the
# Swift Technical Handbook
# <https://swift.gsfc.nasa.gov/proposals/tech_appd/swiftta_v14/node24.html>.
visibility_constraints = [
    # SAA constraint
    OutsideSouthAtlanticAnomalyConstraint(healpix.nside),
    # 28° from the Earth's limb
    EarthLimbConstraint(28 * u.deg),
    # 46° from the Sun
    astroplan.SunSeparationConstraint(46 * u.deg),
    # 23° from the Moon
    astroplan.MoonSeparationConstraint(23 * u.deg),
    # 10° from Galactic plane
    astroplan.GalacticLatitudeConstraint(10 * u.deg)
]
