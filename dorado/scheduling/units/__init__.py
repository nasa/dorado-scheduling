#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""
This module introduces an `orbit` Astropy unit, which allows you to express
time intervals in units of an orbital period.

Examples
--------

First, some imports::

>>> from astropy import units as u
>>> from dorado.scheduling import data
>>> from dorado.scheduling import TLE
>>> from importlib import resources

Load an example two-line element:

>>> with resources.path(data, 'dorado-625km-sunsync.tle') as p:
...     orbit = TLE(p)

Now convert from units of the orbital period to units of seconds:

>>> u.Quantity('2 orbit').to(u.s, equivalencies=equivalencies.orbital(orbit))
<Quantity 11664.87018328 s>
"""

from .orbital import orbit
from . import equivalencies

__all__ = ('equivalencies', 'orbit')
