import numpy as np
from scipy.interpolate import interpolate as interp


def uniformSphere(npoints, seed=42):
    """
    Just make RA, dec points on a sphere
    """
    np.random.seed(seed)
    u = np.random.uniform(size=npoints)
    v = np.random.uniform(size=npoints)

    ra = 2.*np.pi * u
    dec = np.arccos(2.*v - 1.)
    # astro convention of -90 to 90
    dec -= np.pi/2.
    return np.degrees(ra), np.degrees(dec)


def calc_lc(tini, tmax, dt, mej, vej, beta, kappa_r):

    # ** define constants **
    c = 3.0e10
    # mp = 1.67e-24
    Msun = 2.0e33
    kb = 1.38e-16
    sigSB = 5.67e-5
    h = 6.63e-27
    arad = 7.56e-15
    Mpc = 3.08e24

    # ** define parameters **

    # fiducial redshift/distance
    # z = 0.01
    # D = 39.5*Mpc
    z = 0.00
    D = 1e-5*Mpc

    # define desired observer band wavelengths (nm)
    # u (0), b (1), v (2), r (3), i (4), z (5), y(6), j (7), k (8), l (9)
    # lambdaobs = np.array([365., 445., 551., 658., 806., 900.,
    #                       1020., 1220., 2190., 3450.])

    # FUV (0) NUV (1)
    lambdaobs = np.array([160.0, 250.0])

    nuobs = c/(1.0e-7*lambdaobs)
    nuobs = nuobs/(1.0 + z)

    # total ejecta mass
    M0 = mej*Msun
    # minimum initial velocity
    v0 = vej*c
    # velocity index (M ~ v**-beta)
    # beta = 3.
    # initial thermal energy of bulk
    E0 = (M0)*(v0**2.0)/2.0
    # normalization of opacity of r-process matter
    # (~10 for lanthanides, ~1 for non-lanthanides)
    # kappa_r = 10.
    # IGNORE PARAMETERS BELOW THIS LINE
    # mass cut of free neutrons
    Mn = 1.0e-8*Msun
    # electron fraction & initial neutron mass fraction in outermost layers
    Ye = 0.1
    Xn0max = 1.0-2.0*Ye
    # engine (0 = off, 1 = on)
    engine_switch = 0
    # BH (0 = magnetar, 1 = BH)
    BH_switch = 0
    ej = 0.1
    # magnetar period (in seconds) and magnetic field (G)
    P = 0.7e-3
    B = 1.0e15
    # magnetar collapse time (in units of initial spin-down times)
    tcollapse = 10000000.

    # ** define time array in seconds **
    # tprec = 10000
    # tmin = np.log(0.1)
    # tmax = np.log(1.0e6)
    # t = np.arange(tprec)*(tmax-tmin)/(tprec-1.0) + tmin
    # t = np.exp(t)
    # tdays = t/(3600.*24.)

    tdays = np.arange(tini, tmax+dt, dt)
    t = tdays*(3600.*24.)
    tprec = len(t)

    # ** define mass/velocity array of outer ejecta,
    # comprised of half of mass **
    mmin = np.log(1.0e-8)
    mmax = np.log(M0/Msun)
    mprec = 300
    m = np.arange(mprec)*(mmax-mmin)/(mprec-1.0) + mmin
    m = np.exp(m)

    # vm(where(m gt 0.5*M0/Msun)) = v0
    # vm(where(m le 0.5*M0/Msun)) =
    # v0*(m(where(m le 0.5*M0/Msun))/(0.5*M0/Msun))^(-1./beta)
    vm = v0*(m/(M0/Msun))**(-1./beta)
    vm[vm > c] = c

    # define thermalization efficiency from Barnes+16
    # 1e-2 Msun, 0.2 c
    # ca3 = 1.3
    # cb3 = 0.2
    # cd3 = 1.1
    # 1e-3, 0.3 c
    # ca2 = 8.2
    # cb2 = 1.2
    # cd2 = 1.52
    # 1e-2, 0.1 c
    ca = 0.56
    cb = 0.17
    cd = 0.74
    eth = 0.36*(np.exp(-ca*tdays) +
                np.log(1.0+2*cb*(tdays**(cd)))/(2*cb*tdays**(cd)))
    # eth2 = 0.36*(np.exp(-ca2*tdays) +
    #              np.log(1.0+2*cb2*(tdays**(cd2)))/(2*cb2*tdays**(cd2)))
    # eth3 = 0.36*(np.exp(-ca3*tdays) +
    #              np.log(1.0+2*cb3*(tdays**(cd3)))/(2*cb3*tdays**(cd3)))

    # ** calculate magnetar power **
    Rns = 12.e5
    # moment of inertia
    Ins = 1.3e20
    Ins = Ins*1.0e25
    # magnetic moment
    mu = B*(Rns**(3.0))
    # angular rotation rate
    omega = 2.0*np.pi/P
    # rotational energy
    Erot = 0.5*Ins*omega**(2.0)
    # maximum spin-down luminosity
    Lsd0 = mu**(2.0)*(omega**(4.0))/c**(3.0)
    tsd0 = Erot/Lsd0
    Lsd = Lsd0/(1.0 + t/tsd0)**(2.0)
    Lsd[t > tcollapse*tsd0] = 0.0
    Lsd = Lsd/1.0e20
    Lsd = Lsd/1.0e20
    # Lsd2 = Lsd

    if BH_switch:
        # *** calculate BH fall-back power
        Lsd = 2.0e11*(ej/0.1)*(t/0.1)**(-5./3.)
    if not engine_switch:
        Lsd[:] = 0.0

    # ** define diffusive mass depth (assumed beta = 3) **
    Mdiff = (4.0*np.pi*(M0)**(1./3.)*(v0*c*t**2.)/(3.0*kappa_r))**(3./4.)
    Mdiff[Mdiff > M0] = M0
    Mdiff = Mdiff/Msun

    # ** define radioactive heating rates **
    # neutron and r-process mass fractions
    Xn0 = Xn0max*2*np.arctan((Mn/(m*Msun))**(1.0))/np.pi
    Xr = 1.0-Xn0

    # define arrays in mass layer and time
    Xn = np.zeros((mprec, tprec))
    edotn = np.zeros((mprec, tprec))
    edotr = np.zeros((mprec, tprec))
    edot = np.zeros((mprec, tprec))
    kappa = np.zeros((mprec, tprec))
    kappan = np.zeros((mprec, tprec))
    kappar = np.zeros((mprec, tprec))

    # define specific heating rates and opacity of each mass layer
    t0 = 1.3
    sig = 0.11

    tarray = np.tile(t, (mprec, 1))
    Xn0array = np.tile(Xn0, (tprec, 1)).T
    Xrarray = np.tile(Xr, (tprec, 1)).T
    etharray = np.tile(eth, (mprec, 1))
    Xn = Xn0array*np.exp(-tarray/900.)
    edotn = 3.2e14*Xn
    edotr = 4.0e18*Xrarray*(0.5 - (1./np.pi) *
                            np.arctan((tarray-t0)/sig))**(1.3)*etharray
    edotr = 2.1e10*etharray*((tarray/(3600.*24.))**(-1.3))
    edot = edotn + edotr
    kappan = 0.4*(1.0-Xn-Xrarray)
    kappar = kappa_r*Xrarray
    kappa = kappan + kappar

    # define total r-process heating of inner layer
    Lr = M0*4.0e18*(0.5 - (1./np.pi)*np.arctan((t-t0)/sig))**(1.3)*eth
    Lr = Lr/1.0e20
    Lr = Lr/1.0e20

    # *** define arrays by mass layer/time arrays ***
    ene = np.zeros((mprec, tprec))
    lum = np.zeros((mprec, tprec))
    # lumpdv = np.zeros((mprec, tprec))
    # lumedot = np.zeros((mprec, tprec))
    tdiff = np.zeros((mprec, tprec))
    tau = np.zeros((mprec, tprec))
    # properties of photosphere
    Rphoto = np.zeros((tprec, ))
    vphoto = np.zeros((tprec, ))
    mphoto = np.zeros((tprec, ))
    kappaphoto = np.zeros((tprec, ))

    # *** define arrays for total ejecta (1 zone = deepest layer) ***
    # thermal energy
    E = np.zeros((tprec, ))
    # kinetic energy
    Ek = np.zeros((tprec,))
    # velocity
    v = np.zeros((tprec, ))
    R = np.zeros((tprec, ))
    taues = np.zeros((tprec, ))
    Lrad = np.zeros((tprec, ))
    temp = np.zeros((tprec, ))
    # setting initial conditions
    E[0] = E0/1.0e20
    E[0] = E[0]/1.0e20
    Ek[0] = E0/1.0e20
    Ek[0] = Ek[0]/1.0e20
    v[0] = v0
    R[0] = t[0]*v[0]

    dt = t[1:]-t[:-1]
    dm = m[1:]-m[:-1]
    # marray = np.tile(m, (tprec, 1)).T
    # dmarray = np.tile(dm, (tprec, 1)).T

    for j in range(tprec-1):
        # one zone calculation
        temp[j] = 1.0e10*(3.0*E[j]/(arad*4.0*np.pi*R[j]**(3.0)))**(0.25)
        if (temp[j] > 4000.):
            kappaoz = kappa_r
        if (temp[j] < 4000.):
            kappaoz = kappa_r*(temp[j]/4000.)**(5.5)
        kappaoz = kappa_r
        LPdV = E[j]*v[j]/R[j]
        tdiff0 = 3.0*kappaoz*M0/(4.0*np.pi*c*v[j]*t[j])
        tlc0 = R[j]/c
        tdiff0 = tdiff0+tlc0
        Lrad[j] = E[j]/tdiff0
        Ek[j+1] = Ek[j] + LPdV*(dt[j])
        v[j+1] = 1.0e20*(2.0*Ek[j]/(M0))**(0.5)
        E[j+1] = (Lr[j] + Lsd[j]-LPdV-Lrad[j])*(dt[j]) + E[j]
        R[j+1] = v[j+1]*(dt[j]) + R[j]
        taues[j+1] = (M0)*0.4/(4.0*R[j+1]**(2.0))

        templayer = (3.0*ene[:-1, j]*dm*Msun /
                     (arad*4.0*np.pi*(t[j]*vm[:-1])**(3.0)))**(0.25)
        kappa_correction = np.ones(templayer.shape)
        kappa_correction[templayer > 4000.] = 1.0
        kappa_correction[templayer < 4000.] = (templayer[templayer < 4000.] /
                                               4000.)**(5.5)
        kappa_correction[:] = 1.0

        tdiff[:-1, j] = 0.08*kappa[:-1, j]*m[:-1]*Msun*3*kappa_correction / \
            (vm[:-1]*c*t[j]*beta)
        tau[:-1, j] = m[:-1]*Msun*kappa[:-1, j] / \
            (4.0*np.pi*(t[j]*vm[:-1])**(2.0))
        lum[:-1, j] = ene[:-1, j]/(tdiff[:-1, j] + t[j]*(vm[:-1]/c))
        ene[:-1, j+1] = (edot[:-1, j] - (ene[:-1, j]/t[j]) - lum[:-1, j]) * \
            (dt[j]) + ene[:-1, j]
        lum[:-1, j] = lum[:-1, j]*(dm)*Msun

        tau[mprec-1, j] = tau[mprec-2, j]
        # photosphere
        # pig1 = np.argmin(np.abs(tdiff[:, j]-t[j]))
        pig = np.argmin(np.abs(tau[:, j]-1.0))
        vphoto[j] = vm[pig]
        Rphoto[j] = vphoto[j]*t[j]
        mphoto[j] = m[pig]
        kappaphoto[j] = kappa[pig, j]

    Ltotm = np.sum(lum, axis=0)
    Ltotm = Ltotm/1.0e20
    Ltotm = Ltotm/1.0e20

    if engine_switch:
        Ltot = Lrad
        Tobs = 1.0e10*(Ltot/(4.0*np.pi*(R)**(2.0)*sigSB))**(0.25)
        if not BH_switch:
            tlife = (Lsd/1.0e5)**(0.5)*(v/(0.3*c))**(0.5) * \
                (t/(3600.*24.))**(-0.5)
            Ltot = Ltot/(1.0+tlife)
    if not engine_switch:
        Ltot = Ltotm
        Tobs = 1.0e10*(Ltot/(4.0*np.pi*(Rphoto)**(2.0)*sigSB))**(0.25)

    nuobsarray = np.tile(nuobs, (tprec, 1)).T
    expo = np.exp(h*nuobsarray/(kb*Tobs))-1.0
    F = (2.0*np.pi*(h*nuobsarray)*((nuobsarray/c)**(2.0))/expo) * \
        (Rphoto/D)*(Rphoto/D)

    mAB = -2.5*np.log10(F) - 48.6

    # distance modulus
    # muD = 5.0*np.log10(D/(3.08e18))-5.

    return tdays, Ltotm*1e40, mAB, Tobs


class KN_lc(object):
    """
    Calculate some KNe lightcurves

    Parameters
    ----------
    file_list : list of str (None)
        List of file paths to load.
        If None, loads up all the files from data/tde/
    """

    def __init__(self, mejs, vejs, betas, kappas):

        filts = ["FUV", "NUV"]
        magidxs = [0, 1]

        tini, tmax, dt = 0.05, 3.0, 0.1

        # Let's organize the data in to a list of dicts for easy lookup
        self.data = []
        for mej, vej, beta, kappa_r in zip(mejs, vejs, betas, kappas):
            t, lbol, mag_ds, Tobs = calc_lc(tini, tmax, dt,
                                            mej, vej, beta, kappa_r)
            new_dict = {}
            for ii, (filt, magidx) in enumerate(zip(filts, magidxs)):
                jj = np.where(np.isfinite(mag_ds[magidx, :]))[0]
                f = interp.interp1d(t[jj], mag_ds[magidx, jj],
                                    fill_value='extrapolate')
                new_dict[filt] = {'ph': t, 'mag': f(t)}
            self.data.append(new_dict)

    def interp(self, t, filtername, lc_indx=0):
        """
        t : array of floats
            The times to interpolate the light curve to.
        filtername : str
            The filter. one of ugrizy
        lc_index : int (0)
            Which file to use.
        """

        result = np.interp(t.jd, self.data[lc_indx][filtername]['ph'],
                           self.data[lc_indx][filtername]['mag'],
                           left=99, right=99)
        return result


class KNePopMetric:
    def __init__(self, mejs, vejs, betas, kappas,
                 m5Col='limmag', filterCol='filter'):

        self.filterCol = filterCol
        self.m5Col = m5Col

        self.lightcurves = KN_lc(mejs, vejs, betas, kappas)
        waves = {'FUV': 160., 'NUV': 250.}
        self.waves = waves
        self.R_v = 3.1

    def _single_detect(self, dataSlice, slicePoint, mags, t):
        """
        Simple detection criteria: detect at least once
        """
        result = 1
        # detected in at least two bands
        around_peak = np.where((t > 0) & (t < 30) &
                               (mags < dataSlice[self.m5Col]))[0]
        filters = dataSlice[self.filterCol][around_peak]
        if np.size(filters) < 1:
            return 0

        return result

    def _multi_detect(self, dataSlice, slicePoint, mags, t):
        """
        Simple detection criteria: detect at least twice
        """
        result = 1
        # detected in at least two bands
        around_peak = np.where((t > 0) & (t < 30) &
                               (mags < dataSlice[self.m5Col]))[0]
        filters = dataSlice[self.filterCol][around_peak]
        if np.size(filters) < 2:
            return 0

        return result

    def _multi_color_detect(self, dataSlice, slicePoint, mags, t):
        """
        Color-based simple detection criteria:
        detect at least twice, with at least two color
        """
        result = 1
        # detected in at least two bands
        around_peak = np.where((t > 0) & (t < 30) &
                               (mags < dataSlice[self.m5Col]))[0]
        filters = np.unique(dataSlice[self.filterCol][around_peak])
        if np.size(filters) < 2:
            return 0

        return result

    def run(self, dataSlice, slicePoint=None):
        result = {}
        t = dataSlice["time"] - slicePoint['peak_time']
        mags = np.zeros(t.size, dtype=float)
        lc_indx = slicePoint['file_indx']

        for filtername in np.unique(dataSlice[self.filterCol]):
            infilt = np.where(dataSlice[self.filterCol] == filtername)
            mags[infilt] = self.lightcurves.interp(t[infilt],
                                                   filtername,
                                                   lc_indx=lc_indx)
            # Apply dust extinction on the light curve
            # A_x = (self.a[filtername][0]+self.b[filtername][0]/self.R_v)*
            # (self.R_v*slicePoint['ebv'])
            # mags[infilt] -= A_x

            distmod = 5*np.log10(slicePoint['distance']*1e6) - 5.0
            mags[infilt] += distmod

        result['single_detect'] = self._single_detect(dataSlice, slicePoint,
                                                      mags, t.jd)
        result['multi_detect'] = self._multi_detect(dataSlice, slicePoint,
                                                    mags, t.jd)
        result['multi_color_detect'] = self._multi_color_detect(dataSlice,
                                                                slicePoint,
                                                                mags, t.jd)

        return result

    def reduce_single_detect(self, metric):
        return metric['single_detect']

    def reduce_multi_detect(self, metric):
        return metric['multi_detect']

    def reduce_multi_color_detect(self, metric):
        return metric['multi_color_detect']


def generateKNPopSlicer(t_start=1, t_end=3652, n_events=10000,
                        seed=42, n_files=100):
    """ Generate a population of KNe events, and put the info
        about them into a UserPointSlicer object

    Parameters
    ----------
    t_start : float (1)
        The night to start tde events on (days)
    t_end : float (3652)
        The final night of TDE events
    n_events : int (10000)
        The number of TDE events to generate
    seed : float
        The seed passed to np.random
    n_files : int (7)
        The number of different TDE lightcurves to use
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x) \\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    ra, dec = uniformSphere(n_events, seed=seed)
    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0,
                                           high=n_files,
                                           size=n_events)).astype(int)
    distance = rndm(10, 300, 4, size=n_events)

    slicer = []
    for r, d, p, f, i in zip(ra, dec, peak_times, file_indx, distance):
        slicer.append({'ra': r,
                       'dec': d,
                       'peak_time': p,
                       'file_indx': f,
                       'distance': i})

    return slicer
