#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from astropy.units import def_unit

_ns = globals()

def_unit('orbit', namespace=_ns,
         doc='a unit of time equal to one orbital period')

del _ns, def_unit


def _enable():
    import inspect
    from astropy.units import add_enabled_units
    add_enabled_units(inspect.getmodule(_enable))


_enable()
