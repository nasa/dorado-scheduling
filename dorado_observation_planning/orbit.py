from importlib import resources

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import skyfield.api

from . import data

__all__ = ('position',)


# Load two-line element for satellite (use Swift's orbit)
with resources.path(data, 'orbits.txt') as path:
    TLE = skyfield.api.load.tle(str(path))['SWIFT']

TIMESCALE = skyfield.api.load.timescale()


def position(time):
    """Get the position of the satellite.

    Parameters
    ----------
    satellite : skyfield.sgp4lib.EarthSatellite
        An Earth satellite model from Skyfield.
    time : astropy.time.Time, skyfield.timelib.Timeto
        The time of the observation.

    Returns
    -------
    earth_location : astropy.coordinates.EarthLocation
        The geocentric position of the satellite.
    """
    if isinstance(time, Time):
        time = TIMESCALE.from_astropy(time)
    return EarthLocation.from_geocentric(*TLE.at(time).position.to(u.meter))
