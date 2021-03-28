#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Astroplan visibility constraints."""
import astroplan
from astropy import units as u

from .earth_limb import EarthLimbConstraint
from .orbit_night import OrbitNightConstraint
from .radiation import TrappedParticleFluxConstraint
from ..skygrid import healpix

__all__ = ('visibility_constraints',
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
