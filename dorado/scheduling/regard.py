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


def get_field_of_regard(times):
    return np.asarray(list(progress_map(
        _observable, times, orbit.get_position(times), jobs=None)))
