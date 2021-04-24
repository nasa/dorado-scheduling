"""
Filling factor of various sky grids for the Dorado mission
==========================================================
"""

# %%
# First, some imports.

from functools import partial

from astropy_healpix import HEALPix
from astropy.coordinates import ICRS
from astropy import units as u
from dorado.scheduling import mission, skygrid
from matplotlib import pyplot as plt
import numpy as np

# %%
# Select a HEALPix resolution for spatial calculations.

hpx = HEALPix(nside=512, frame=ICRS())

# %%
# Determine the area of the field of view.

fov = mission.dorado.fov
fov_area = len(fov.footprint_healpix(hpx)) * hpx.pixel_area

# %%
# Compute the filling factor for all sky grid methods.

methods = {
    'geodesic(class_="I")': partial(skygrid.geodesic, class_='I'),
    'geodesic(class_="II")': partial(skygrid.geodesic, class_='II'),
    'geodesic(class_="III")': partial(skygrid.geodesic, class_='III'),
    'golden_angle_spiral': skygrid.golden_angle_spiral,
    'sinusoidal': skygrid.sinusoidal
}

ax = plt.axes()
ax.set_xlabel('Number of tiles')
ax.set_ylabel('1 - filling factor')
ax.set_yscale('log')

areas = np.geomspace(0.5 * fov_area, fov_area)
for method_name, method in methods.items():
    number_of_tiles = np.empty(len(areas), dtype=int)
    filling_factors = np.empty(len(areas), dtype=float)
    for i, area in enumerate(areas):
        centers = method(area)
        rolls = np.asarray([0]) * u.deg
        pixels = set()
        for more_pixels, in fov.footprint_healpix_grid(hpx, centers, rolls):
            pixels |= set(more_pixels)
        number_of_tiles[i] = len(centers)
        filling_factors[i] = len(pixels) / hpx.npix

    ax.plot(number_of_tiles, 1 - filling_factors, '.-', label=method_name)

ax.legend()
