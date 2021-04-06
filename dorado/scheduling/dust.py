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
        bandpassDict = {'FUV': [1350, 1750],
                        'NUV': [1750, 2800]}
        zeropointDict = {'FUV': 22.0,
                         'NUV': 23.5}
        self.sb = None
        self.phi = None

        for filtername in bandpassDict:
            wavelen_min = bandpassDict[filtername][0]
            wavelen_max = bandpassDict[filtername][1]
            self.setFlatSED(wavelen_min=wavelen_min,
                            wavelen_max=wavelen_max,
                            wavelen_step=1.0)
            self.sb = np.zeros(len(self.wavelen), dtype='float')
            self.sb[(self.wavelen >= wavelen_min) &
                    (self.wavelen <= wavelen_max)] = 1.0

            self.ref_ebv = ref_ebv
            self.zp = zeropointDict[filtername]
            # Calculate non-dust-extincted magnitude
            flatmag = self.calcMag()
            # Add dust
            a, b = self.setupCCM_ab()
            self.addDust(a, b, ebv=self.ref_ebv, R_v=R_v)
            # Calculate difference due to dust when EBV=1.0
            # (m_dust = m_nodust - Ax, Ax > 0)
            self.Ax1[filtername] = self.calcMag() - flatmag

    def setupCCM_ab(self, wavelen=None):
        """
        Calculate a(x) and b(x) for CCM dust model. (x=1/wavelen).

        If wavelen not specified, calculates a and b on the own object's
        wavelength grid.
        Returns a(x) and b(x) can be common to many seds, wavelen is the same.

        This method sets up extinction due to the model of
        Cardelli, Clayton and Mathis 1989 (ApJ 345, 245)
        """
        # This extinction law taken from Cardelli, Clayton and Mathis ApJ 1989.
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns), then different values for a(x) and
        # b(x) depending on wavelength regime.
        # Also, the extinction is parametrized as R_v = A_v / E(B-V).
        # Magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        if wavelen is None:
            wavelen = np.copy(self.wavelen)
        a_x = np.zeros(len(wavelen), dtype='float')
        b_x = np.zeros(len(wavelen), dtype='float')
        # Convert wavelength to x (in inverse microns).
        x = np.empty(len(wavelen), dtype=float)
        nm_to_micron = 1/1000.0
        x = 1.0 / (wavelen * nm_to_micron)
        # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
        condition = (x >= 0.3) & (x <= 1.1)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 0.574 * y**1.61
            b_x[condition] = -0.527 * y**1.61
        # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
        condition = (x >= 1.1) & (x <= 3.3)
        if len(a_x[condition]) > 0:
            y = x[condition] - 1.82
            a_x[condition] = 1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3 +\
                0.72085*y**4
            a_x[condition] = a_x[condition] + 0.01979*y**5 - 0.77530*y**6 +\
                0.32999*y**7
            b_x[condition] = 1.41338*y + 2.28305*y**2 + 1.07233*y**3 -\
                5.38434*y**4
            b_x[condition] = b_x[condition] - 0.62251*y**5 + 5.30260*y**6 -\
                2.09002*y**7
        # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
        condition = (x >= 3.3) & (x < 5.9)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
            b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        condition = (x > 5.9) & (x < 8)
        if len(a_x[condition]) > 0:
            y = x[condition]
            Fa_x = np.empty(len(a_x[condition]), dtype=float)
            Fb_x = np.empty(len(a_x[condition]), dtype=float)
            Fa_x = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
            Fb_x = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
            a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341) +\
                Fa_x
            b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263) +\
                Fb_x
        # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
        condition = (x >= 8) & (x <= 11.)
        if len(a_x[condition]) > 0:
            y = x[condition]-8.0
            a_x[condition] = -1.073 - 0.628*(y) + 0.137*(y)**2 - 0.070*(y)**3
            b_x[condition] = 13.670 + 4.257*(y) - 0.420*(y)**2 + 0.374*(y)**3
        return a_x, b_x

    def setupODonnell_ab(self, wavelen=None):
        """
        Calculate a(x) and b(x) for O'Donnell dust model. (x=1/wavelen).

        If wavelen not specified, calculates a and b on the own object's
        wavelength grid. Returns a(x) and b(x) can be common to many seds,
        wavelen is the same.

        This method sets up the extinction parameters from the model of
        O'Donnel 1994 (ApJ 422, 158)
        """
        # The general form is A_l / A(V) = a(x) + b(x)/R_V
        # (where x=1/lambda in microns), then different values for a(x) and
        # b(x) depending on wavelength regime.
        # Also, the extinction is parametrized as R_v = A_v / E(B-V).
        # Magnitudes of extinction (A_l) translates to flux by
        # a_l = -2.5log(f_red / f_nonred).
        if wavelen is None:
            wavelen = np.copy(self.wavelen)
        a_x = np.zeros(len(wavelen), dtype='float')
        b_x = np.zeros(len(wavelen), dtype='float')
        # Convert wavelength to x (in inverse microns).
        x = np.empty(len(wavelen), dtype=float)
        nm_to_micron = 1/1000.0
        x = 1.0 / (wavelen * nm_to_micron)
        # Dust in infrared 0.3 /mu < x < 1.1 /mu (inverse microns).
        condition = (x >= 0.3) & (x <= 1.1)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 0.574 * y**1.61
            b_x[condition] = -0.527 * y**1.61
        # Dust in optical/NIR 1.1 /mu < x < 3.3 /mu region.
        condition = (x >= 1.1) & (x <= 3.3)
        if len(a_x[condition]) > 0:
            y = x[condition] - 1.82
            a_x[condition] = 1 + 0.104*y - 0.609*y**2 + 0.701*y**3 + 1.137*y**4
            a_x[condition] = a_x[condition] - 1.718*y**5 - 0.827*y**6 +\
                1.647*y**7 - 0.505*y**8
            b_x[condition] = 1.952*y + 2.908*y**2 - 3.989*y**3 - 7.985*y**4
            b_x[condition] = b_x[condition] + 11.102*y**5 + 5.491*y**6 -\
                10.805*y**7 + 3.347*y**8
        # Dust in ultraviolet and UV (if needed for high-z) 3.3 /mu< x< 8 /mu.
        condition = (x >= 3.3) & (x < 5.9)
        if len(a_x[condition]) > 0:
            y = x[condition]
            a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341)
            b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263)
        condition = (x > 5.9) & (x < 8)
        if len(a_x[condition]) > 0:
            y = x[condition]
            Fa_x = np.empty(len(a_x[condition]), dtype=float)
            Fb_x = np.empty(len(a_x[condition]), dtype=float)
            Fa_x = -0.04473*(y-5.9)**2 - 0.009779*(y-5.9)**3
            Fb_x = 0.2130*(y-5.9)**2 + 0.1207*(y-5.9)**3
            a_x[condition] = 1.752 - 0.316*y - 0.104/((y-4.67)**2 + 0.341) +\
                Fa_x
            b_x[condition] = -3.090 + 1.825*y + 1.206/((y-4.62)**2 + 0.263) +\
                Fb_x
        # Dust in far UV (if needed for high-z) 8 /mu < x < 10 /mu region.
        condition = (x >= 8) & (x <= 11.)
        if len(a_x[condition]) > 0:
            y = x[condition]-8.0
            a_x[condition] = -1.073 - 0.628*(y) + 0.137*(y)**2 - 0.070*(y)**3
            b_x[condition] = 13.670 + 4.257*(y) - 0.420*(y)**2 + 0.374*(y)**3
        return a_x, b_x

    def addDust(self, a_x, b_x, A_v=None, ebv=None, R_v=3.1,
                wavelen=None, flambda=None):
        """
        Add dust model extinction to the SED, modifying flambda and fnu.

        Get a_x and b_x either from setupCCMab or setupODonnell_ab

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

        A_lambda = (a_x + b_x / R_v) * A_v
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
