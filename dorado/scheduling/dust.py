#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""
Dust map infrastructure. Software modified from VRO's photUtils.
https://github.com/lsst/sims_photUtils/tree/master/python/lsst/sims/photUtil
"""

import numpy as np
import astropy.units as u

from dust_extinction.parameter_averages import CCM89
from synphot import ReddeningLaw
from synphot import SourceSpectrum, SpectralElement
from synphot.models import ConstFlux1D, Box1D


class Dust:
    """Calculate extinction values

    Parameters
    ----------
    config: simsurvey config file
    R_v : float (3.1)
        Extinction law parameter (3.1).
    ref_ev : float (1.)
        The reference E(B-V) value to use. Things in MAF assume 1.
    """
    def __init__(self, filters=['FUV', 'NUV'],
                 bandpasses=[[1350, 1750], [1750, 2800]],
                 zeropoints=[22.0, 23.5],
                 R_v=3.1, ref_ebv=1.):
        # Calculate dust extinction values
        self.Ax1 = {}
        self.bandpassDict = {}
        self.zeropointDict = {}

        for ii, filt in enumerate(filters):
            self.bandpassDict[filt] = bandpasses[ii]
            self.zeropointDict[filt] = zeropoints[ii]

        redlaw = ReddeningLaw(CCM89(Rv=R_v))
        for filtername in self.bandpassDict:
            wavelen_min = self.bandpassDict[filtername][0]
            wavelen_max = self.bandpassDict[filtername][1]
            wav = np.arange(wavelen_min, wavelen_max, 1.0) * u.AA
            flat_abmag = SourceSpectrum(ConstFlux1D, amplitude=0*u.STmag)
            bp = SpectralElement(Box1D,
                                 amplitude=1,
                                 x_0=(wavelen_max+wavelen_min)/2.0,
                                 width=wavelen_max-wavelen_min)
            extcurve = redlaw.extinction_curve(ref_ebv, wavelengths=wav)
            sp_ext = flat_abmag * bp * extcurve
            sp = flat_abmag * bp
            sp_ext_mag = -2.5*np.log10(sp_ext.integrate().to_value())
            sp_mag = -2.5*np.log10(sp.integrate().to_value())

            # Calculate difference due to dust when EBV=1.0
            # (m_dust = m_nodust - Ax, Ax > 0)
            self.Ax1[filtername] = sp_ext_mag - sp_mag
