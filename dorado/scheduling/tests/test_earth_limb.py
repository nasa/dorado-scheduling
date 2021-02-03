#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astroplan import FixedTarget, Observer
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy import units as u
from hypothesis import given, settings
from hypothesis.strategies import floats

from ..constraints.earth_limb import EarthLimbConstraint


@given(lon=floats(-180, 180), lat=floats(-90, 90),
       ra=floats(0, 360), dec=floats(-90, 90),
       time=floats(51544.5, 51909.75), min=floats(-90, 90))
@settings(deadline=None)
def test_observer_on_surface(lon, lat, ra, dec, time, min):
    """Test an obsever on the surface of the Earth."""
    obstime = Time(time, format='mjd')
    location = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg)
    observer = Observer(location)
    target = FixedTarget(SkyCoord(ra*u.deg, dec*u.deg))
    constraint = EarthLimbConstraint(min * u.deg)
    observable = constraint.compute_constraint(obstime, observer, target)
    alt = observer.altaz(obstime, target).alt

    if alt > (min + 1) * u.deg:
        assert observable
    elif alt < (min - 1) * u.deg:
        assert not observable


@given(lon=floats(-180, 180), lat=floats(-90, 90),
       ra=floats(0, 360), dec=floats(-90, 90),
       time=floats(51544.5, 51909.75), min=floats(-90, 90))
@settings(deadline=None)
def test_low_earth_orbit(lon, lat, ra, dec, time, min):
    """Test an observer in low Earth orbit."""
    obstime = Time(time, format='mjd')
    location = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, 550 * u.km)
    observer = Observer(location)
    target = FixedTarget(SkyCoord(ra*u.deg, dec*u.deg))
    constraint = EarthLimbConstraint(min * u.deg)
    observable = constraint.compute_constraint(obstime, observer, target)
    alt = observer.altaz(obstime, target).alt

    if alt > (min - 22.5) * u.deg:
        assert observable
    elif alt < (min - 23.5) * u.deg:
        assert not observable


@given(lon=floats(-180, 180), lat=floats(-90, 90),
       ra=floats(0, 360), dec=floats(-90, 90),
       time=floats(51544.5, 51909.75), min=floats(-90, 90))
@settings(deadline=None)
def test_distant_obsever(lon, lat, ra, dec, time, min):
    """Test a very distant observer."""
    obstime = Time(time, format='mjd')
    location = EarthLocation.from_geodetic(lon*u.deg, lat*u.deg, 1*u.au)
    observer = Observer(location)
    target = FixedTarget(SkyCoord(ra*u.deg, dec*u.deg))
    constraint = EarthLimbConstraint(min * u.deg)
    observable = constraint.compute_constraint(obstime, observer, target)
    alt = observer.altaz(obstime, target).alt

    if alt > (min - 90) * u.deg:
        assert observable
    elif alt < (min - 90 - 1e-2) * u.deg:
        assert not observable
