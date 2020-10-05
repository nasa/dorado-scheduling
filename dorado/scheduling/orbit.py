"""Spacecraft orbit."""
from importlib import resources

from astropy.coordinates import EarthLocation
from astropy.time import Time
from astropy import units as u
import skyfield.api

from . import data

__all__ = ('get_position',)


# Load two-line element for satellite (use Swift's orbit)
with resources.path(data, 'orbits.txt') as path:
    satellite = skyfield.api.load.tle(str(path))['SWIFT']

timescale = skyfield.api.load.timescale()


def get_position(time):
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
        time = timescale.from_astropy(time)
    position = satellite.at(time).position
    return EarthLocation.from_geocentric(*position.to(u.meter))
