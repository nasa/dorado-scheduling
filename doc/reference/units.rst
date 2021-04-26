Units (`dorado.scheduling.units`)
=================================
.. automodule:: dorado.scheduling.units

Examples
--------

First, some imports:

>>> from astropy import units as u
>>> from dorado.scheduling import data
>>> from dorado.scheduling import Orbit
>>> from dorado.scheduling.units import equivalencies
>>> from importlib import resources

Load an example two-line element:

>>> with resources.path(data, 'dorado-625km-sunsync.tle') as p:
...     orbit = Orbit(p)

>>> u.Quantity('2 orbit').to(u.s, equivalencies=equivalencies.orbital(orbit))
