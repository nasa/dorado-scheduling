#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.units.equivalencies import Equivalency

from .orbital import orbit as _orbit


def orbital(orbit):
    return Equivalency([(_orbit, orbit.period.unit,
                         lambda x: x * orbit.period.value,
                         lambda x: x / orbit.period.value)],
                       name='orbital')
