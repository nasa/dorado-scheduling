#
# Copyright © 2020 United States Government as represented by the Administrator
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

from dustmaps.planck import PlanckQuery
import extinction
planck = PlanckQuery()

# speed of light
lightspeed = 299792458.0
# planck's constant
planck = 6.626068e-27
# nanometers to meters conversion = 1e-9 m/nm
nm2m = 1.00e-9
# erg/cm2/s/Hz to Jansky units (fnu)
ergsetc2jansky = 1.00e23


class Dust():
    """Calculate extinction values

    Parameters
    ----------
    R_v : float (3.1)
        Extinction law parameter (3.1).
    ref_ev : float (1.)
        The reference E(B-V) value to use. Things in MAF assume 1.
    """
    def __init__(self, R_v=3.1, ref_ebv=1.):
        # Calculate dust extinction values
        self.Ax1 = {}
        self.bandpassDict = {'FUV': [1350, 1750],
                             'NUV': [1750, 2800]}
        self.zeropointDict = {'FUV': 22.0,
                              'NUV': 23.5}
        self.sb = None
        self.phi = None

        for filtername in self.bandpassDict:
            wavelen_min = self.bandpassDict[filtername][0]
            wavelen_max = self.bandpassDict[filtername][1]
            self.setFlatSED(wavelen_min=wavelen_min,
                            wavelen_max=wavelen_max,
                            wavelen_step=1.0)
            self.sb = np.zeros(len(self.wavelen), dtype='float')
            self.sb[(self.wavelen >= wavelen_min) &
                    (self.wavelen <= wavelen_max)] = 1.0

            self.ref_ebv = ref_ebv
            self.zp = self.zeropointDict[filtername]
            # Calculate non-dust-extincted magnitude
            flatmag = self.calcMag()
            # Add dust
            self.addDust(ebv=self.ref_ebv, R_v=R_v)

            # Calculate difference due to dust when EBV=1.0
            # (m_dust = m_nodust - Ax, Ax > 0)
            self.Ax1[filtername] = self.calcMag() - flatmag

    def addDust(self, A_v=None, ebv=None, R_v=3.1,
                wavelen=None, flambda=None):
        """
        Add dust model extinction to the SED, modifying flambda and fnu.

        Get A_lambda from extinction package.

        Specify any two of A_V, E(B-V) or R_V (=3.1 default).
        """
        if not hasattr(self, '_ln10_04'):
            self._ln10_04 = 0.4*np.log(10.0)

        # The extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns).
        # Then, different values for a(x) and b(x) depending on wavelength
        # regime.
        # Also, the extinction is parametrized as R_v = A_v / E(B-V).
        # The magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        flambda = self.flambda
        self.fnu = None
        # Input parameters for reddening can include any of 3 parameters;
        # only 2 are independent.
        # Figure out what parameters were given, and see if self-consistent.
        if R_v == 3.1:
            if A_v is None:
                A_v = R_v * ebv
            elif (A_v is not None) and (ebv is not None):
                # Specified A_v and ebv, so R_v should be nondefault.
                R_v = A_v / ebv
        if (R_v != 3.1):
            if (A_v is not None) and (ebv is not None):
                calcRv = A_v / ebv
                if calcRv != R_v:
                    mess1 = "CCM parametrization expects R_v = A_v / E(B-V);"
                    mess2 = """Please check input values, because values
                             are inconsistent."""
                    raise ValueError(mess1, mess2)
            elif A_v is None:
                A_v = R_v * ebv
        # R_v and A_v values are specified or calculated.
        A_lambda = extinction.ccm89(self.wavelen, A_v, R_v)
        # dmag_red(dust) = -2.5 log10 (f_red / f_nored) : (f_red / f_nored) =
        # 10**-0.4*dmag_red
        dust = np.exp(-A_lambda*self._ln10_04)
        flambda *= dust
        # Update self if required.
        self.flambda = flambda
        return

    def calcFlux(self, wavelen=None, fnu=None):
        """
        Integrate the specific flux density of the object over the normalized
        response curve of a bandpass, giving a flux in Janskys
        (10^-23 ergs/s/cm^2/Hz) through the normalized response curve, as
        detailed in Section 4.1 of the LSST design document LSE-180 and Section
        2.6 of the LSST Science Book (http://ww.lsst.org/scientists/scibook).
        This flux in Janskys (which is usually though of as a unit of specific
        flux density), should be considered a weighted average of the specific
        flux density over the normalized response curve of the bandpass.
        Because we are using the normalized response curve (phi in LSE-180),
        this quantity will depend only on the shape of the response curve,
        not its absolute normalization.

        Note: the way that the normalized response curve has been defined
        (see equation 5 of LSE-180) is appropriate for photon-counting
        detectors, not calorimeters.

        Passed wavelen/fnu arrays will be unchanged, but if uses self will
        check if fnu is set.

        Calculating the AB mag requires the wavelen/fnu pair to be on the same
        grid as bandpass; (temporary values of these are used).
        """
        # Calculate fnu if required.
        if self.fnu is None:
            self.flambdaTofnu()
        wavelen = self.wavelen
        fnu = self.fnu
        self.sbTophi()
        # Calculate flux in bandpass and return this value.
        dlambda = wavelen[1] - wavelen[0]
        flux = (fnu*self.phi).sum() * dlambda
        return flux

    def magFromFlux(self, flux):
        """
        Convert a flux into a magnitude (implies knowledge of the zeropoint,
        which is stored in this class)
        """

        return -2.5*np.log10(flux) - self.zp

    def calcMag(self, wavelen=None, fnu=None):
        """
        Calculate the AB magnitude of an object using the normalized system
        response (phi from Section 4.1 of the LSST design document LSE-180).

        Can pass wavelen/fnu arrays or use self.
        Self or passed wavelen/fnu arrays will be unchanged.
        Calculating the AB mag requires the wavelen/fnu pair to be on the same
        grid as bandpass; (but only temporary values of these are used).
         """
        flux = self.calcFlux(wavelen=wavelen, fnu=fnu)
        if flux < 1e-300:
            raise Exception("This SED has no flux within this bandpass.")
        mag = self.magFromFlux(flux)
        return mag

    def setFlatSED(self, wavelen_min=None,
                   wavelen_max=None,
                   wavelen_step=None, name='Flat'):
        """
        Populate the wavelength/flambda/fnu fields in sed according
        to a flat fnu source.
        """
        self.wavelen = np.arange(wavelen_min,
                                 wavelen_max+wavelen_step,
                                 wavelen_step, dtype='float')
        self.fnu = np.ones(len(self.wavelen), dtype='float') * 3631
        self.fnuToflambda()
        self.name = name
        return

    def fnuToflambda(self, wavelen=None, fnu=None):
        """
        Convert fnu into flambda.

        Assumes fnu in units of Jansky and flambda in ergs/cm^s/s/nm.
        Can act on self or user can give wavelen/fnu and get
        wavelen/flambda returned.
        """
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        wavelen = self.wavelen
        fnu = self.fnu
        # On with the calculation.
        # Calculate flambda.
        flambda = fnu / wavelen / wavelen * lightspeed / nm2m
        flambda = flambda / ergsetc2jansky
        # If updating self, then *all of wavelen/fnu/flambda will be updated.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        self.wavelen = wavelen
        self.flambda = flambda
        self.fnu = fnu
        return

    def sbTophi(self):
        """
        Calculate and set phi - the normalized system response.
        This function only pdates self.phi.
        """
        # The definition of phi = (Sb/wavelength)/\int(Sb/wavelength)dlambda.
        # Due to definition of class, self.sb and self.wavelen are guaranteed
        # equal-gridded.
        dlambda = self.wavelen[1]-self.wavelen[0]
        self.phi = self.sb/self.wavelen
        # Normalize phi so that the integral of phi is 1.
        phisum = self.phi.sum()
        if phisum < 1e-300:
            mess = "Phi is poorly defined (nearly 0) over bandpass range."
            raise Exception(mess)
        norm = phisum * dlambda
        self.phi = self.phi / norm
        return

    def flambdaTofnu(self, wavelen=None, flambda=None):
        """
        Convert flambda into fnu.

        This routine assumes that flambda is in ergs/cm^s/s/nm
        and produces fnu in Jansky.
        Can act on self or user can provide wavelen/flambda
        and get back wavelen/fnu.
        """
        # Change Flamda to Fnu by multiplying Flambda * lambda^2 = Fv
        # Fv dv = Fl dl .. Fv = Fl dl / dv = Fl dl / (dl*c/l/l) = Fl*l*l/c
        wavelen = self.wavelen
        flambda = self.flambda
        self.fnu = None
        # Now on with the calculation.
        # Calculate fnu.
        fnu = flambda * wavelen * wavelen * nm2m / lightspeed
        fnu = fnu * ergsetc2jansky
        # If are using/updating self, then *all* wavelen/flambda/fnu
        # will be gridded.
        # This is so wavelen/fnu AND wavelen/flambda can be kept in sync.
        self.wavelen = wavelen
        self.flambda = flambda
        self.fnu = fnu
        return
