"""
Plot sky grid for the Dorado mission
====================================
"""

# %%
# First, some imports.

from astropy import units as u
from dorado.scheduling import mission, skygrid
import ligo.skymap.plot  # noqa: F401
from matplotlib import pyplot as plt
import numpy as np

# %%
# Determine the area of the field of view.

fov = mission.dorado().fov
centers = skygrid.sinusoidal(50 * u.deg**2)
footprints = fov.footprint(centers).icrs
verts = np.moveaxis(np.stack([footprints.ra.deg, footprints.dec.deg]), 0, -1)

# %%
# Plot all-sky overview and close-ups at celestial equator and poles.

for kw in [dict(projection='astro globe', center='0d 30d'),
           dict(projection='astro zoom', center='0d 0d', radius='30 deg'),
           dict(projection='astro zoom', center='0d 90d', radius='30 deg'),
           dict(projection='astro zoom', center='0d -90d', radius='30 deg')]:
    fig, ax = plt.subplots(subplot_kw=kw)
    ax.grid()
    transform = ax.get_transform('world')
    for v in verts:
        ax.add_patch(plt.Polygon(v, alpha=0.3, transform=transform))
