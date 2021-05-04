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

fig_width, fig_height = plt.rcParams['figure.figsize']
fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(2 * fig_width, fig_height), sharey=True)
ax1.set_xlabel('Number of tiles')
ax2.set_xlabel('Requested area per tile (deg$^2$)')
ax1.set_ylabel('1 - filling factor')
ax1.set_yscale('log')

areas = np.geomspace(0.5 * fov_area, fov_area).to(u.deg**2)
for method_name, method in methods.items():
    number_of_tiles = np.empty(len(areas), dtype=int)
    one_minus_fill_factors = np.empty(len(areas), dtype=float)
    for i, area in enumerate(areas):
        centers = method(area)
        rolls = np.asarray([0]) * u.deg
        pixels = set()
        for more_pixels, in fov.footprint_healpix_grid(hpx, centers, rolls):
            pixels |= set(more_pixels)
        number_of_tiles[i] = len(centers)
        one_minus_fill_factors[i] = 1 - len(pixels) / hpx.npix

    ax1.plot(number_of_tiles, one_minus_fill_factors, '.-', label=method_name)
    ax2.plot(areas, one_minus_fill_factors, '.-', label=method_name)

ax1.legend()
