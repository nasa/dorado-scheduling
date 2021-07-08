"""
Polar horns
===========
"""

# %%
# First, some imports.

from astroplan import Observer
from astropy.coordinates import ITRS
from astropy.time import Time
from astropy import units as u
from astropy_healpix import HEALPix
from dorado.scheduling.constraints import TrappedParticleFluxConstraint
from ligo.skymap import plot  # noqa: F401
from matplotlib import pyplot as plt
import numpy as np

# %%
# Create trapped particle constraint.

constraint = TrappedParticleFluxConstraint(
    flux=100 * u.cm**-2 * u.s**-1,
    energy=1 * u.MeV,
    particle='e',
    solar='max')

# %%
# Evaluate the constraint on a HEALPix grid in ITRS coordinates.

obstime = Time('2021-01-01')
frame = ITRS(obstime=obstime)
healpix = HEALPix(64, frame=frame)
locations = healpix.healpix_to_skycoord(np.arange(healpix.npix))
locations = ITRS(lon=locations.spherical.lon,
                 lat=locations.spherical.lat,
                 distance=1 * u.Rearth + 600 * u.km,
                 representation_type='spherical')
locations = locations.earth_location

constraint_values = np.asarray([
    constraint.compute_constraint(obstime, Observer(location))
    for location in locations])

# %%
# Plot a map of the true/false value of the constraint.

ax = plt.axes(projection='geo mollweide')
ax.imshow_hpx(constraint_values, order='nearest-neighbor', cmap='binary_r')
ax.grid()
