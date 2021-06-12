import numpy as np
from scipy.interpolate import interpolate as interp
import astropy.units as u

from gwemlightcurves.KNModels.io.Me2017 import calc_lc_UV


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
            t, lbol, mag_ds, Tobs = calc_lc_UV(tini, tmax, dt,
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
        around_peak = np.where((t > 0) & (t < 7) &
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
        around_peak = np.where((t > 0) & (t < 7) &
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
        around_peak = np.where((t > 0) & (t < 7) &
                               (mags < dataSlice[self.m5Col]))[0]
        filters = np.unique(dataSlice[self.filterCol][around_peak])
        if np.size(filters) < 2:
            return 0

        return result

    def run(self, dataSlice, slicePoint=None, extinction=None):
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
            if extinction is not None:
                mags[infilt] += extinction[filtername]

            distmod = 5*np.log10(slicePoint['distance']*1e6) - 5.0
            mags[infilt] += distmod
            mags[infilt] = mags[infilt]
        mags = mags * u.ABmag

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
        The night to start KN events on (days)
    t_end : float (3652)
        The final night of KN events
    n_events : int (10000)
        The number of KN events to generate
    seed : float
        The seed passed to np.random
    n_files : int (100)
        The number of different KN lightcurves to use
    """

    def rndm(a, b, g, size=1):
        """Power-law gen for pdf(x) \\propto x^{g-1} for a<=x<=b"""
        r = np.random.random(size=size)
        ag, bg = a**g, b**g
        return (ag + (bg - ag)*r)**(1./g)

    peak_times = np.random.uniform(low=t_start, high=t_end, size=n_events)
    file_indx = np.floor(np.random.uniform(low=0,
                                           high=n_files,
                                           size=n_events)).astype(int)
    distance = rndm(10, 300, 4, size=n_events)

    slicer = []
    for p, f, i in zip(peak_times, file_indx, distance):
        slicer.append({'peak_time': p,
                       'file_indx': f,
                       'distance': i})

    return slicer
