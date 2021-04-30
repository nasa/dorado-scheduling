#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from importlib import resources

from astropy.coordinates import ITRS
from astropy.time import Time
from astropy import units as u
import numpy as np
from skyfield import api as sf
from skyfield.framelib import itrs as sf_itrs
import pytest

from .. import data
from .. import Orbit

# Three test cases for time argument: scalar, vector, 2D array
TIME0 = Time('2020-03-01')
TIME1 = TIME0 + np.linspace(0, 1, 1000) * u.day
dt1, dt2 = np.meshgrid(np.linspace(0, 1, 10) * u.day,
                       np.linspace(2, 30, 20) * u.day)
TIME2 = TIME0 + dt1 + dt2


@pytest.mark.parametrize('time', [TIME0, TIME1, TIME2])
def test_get_posvel_skyfield(time):
    """Test SGP4 orbit propagation against high-level Skyfield interface."""
    with resources.path(data, 'dorado-625km-sunsync.tle') as path:
        orbit = Orbit(path)
        sf_satellite, = sf.load.tle_file(str(path))

    coord = orbit(time)

    assert coord.shape == time.shape
    assert isinstance(coord.frame, ITRS)
    assert np.all(coord.frame.obstime == time)

    sf_timescale = sf.load.timescale()
    sf_time = sf_timescale.from_astropy(time.ravel())
    sf_pos, sf_vel = sf_satellite.at(sf_time).frame_xyz_and_velocity(sf_itrs)

    np.testing.assert_allclose(coord.ravel().cartesian.xyz.to_value(u.m),
                               sf_pos.to(u.m).value,
                               rtol=0, atol=20)
    np.testing.assert_allclose(coord.ravel().velocity.d_xyz.to_value(u.m/u.s),
                               sf_vel.to(u.m/u.s).value,
                               rtol=0, atol=0.02)
