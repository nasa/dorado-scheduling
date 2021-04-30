#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astroplan.constraints import Constraint
from radbelt import get_flux

__all__ = ('TrappedParticleFluxConstraint')


class TrappedParticleFluxConstraint(Constraint):
    """Constrain the flux of charged particles in the Van Allen belt.

    Parameters
    ----------
    flux : :class:`astropy.units.Quantity`
        The maximum flux in units compatible with cm^-2 s^-1.
    energy : :class:`astropy.units.Quantity`
        The minimum energy.
    particle : {'e', 'p'}
        The particle species: 'e' for electrons, 'p' for protons.
    solar : {'min', 'max'}
        The solar activity: solar minimum or solar maximum.
    """

    def __init__(self, flux, energy, particle, solar):
        self.flux = flux
        self.energy = energy
        self.particle = particle
        self.solar = solar

    def compute_constraint(self, times, observer, targets=None):
        return get_flux(observer.location, times, self.energy,
                        self.particle, self.solar) <= self.flux
