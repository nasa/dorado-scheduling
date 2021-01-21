#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Field of regard."""
from astroplan import is_event_observable, Observer
from ligo.skymap.util import progress_map
import numpy as np

from .constraints import visibility_constraints
from . import orbit
from . import skygrid

__all__ = ('get_field_of_regard',)


def _observable(time, location):
    return is_event_observable(
        visibility_constraints,
        Observer(location),
        skygrid.centers,
        time
    ).ravel()


def get_field_of_regard(times, jobs=None):
    return np.asarray(list(progress_map(
        _observable, times, orbit.get_position(times), jobs=jobs)))
