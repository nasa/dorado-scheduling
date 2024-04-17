#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.units.equivalencies import Equivalency
from astropy import units as u

from .orbital import orbit as _orbit


def orbital(orbit):
    return Equivalency([(_orbit, u.min,
                         lambda x: x * orbit.period.to_value(u.min),
                         lambda x: x / orbit.period.to_value(u.min))],
                       name='orbital')
