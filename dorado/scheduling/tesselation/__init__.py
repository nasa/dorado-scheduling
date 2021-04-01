#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Methods for tesselating the sky into survey tiles.

The functions in this module provide a variety of different methods of
selecting points on the unit sphere with approximately uniform density per unit
area. All of thee functions take one required argument, ``area``, which is the
average area per tile. Some (like :meth:`geodesic`) take additional optional
keyword arguments.

.. autosummary::
    geodesic
    golden_angle_spiral
    sinusoidal

Example
-------
>>> from astropy import units as u
>>> from dorado.scheduling import tesselation
>>> points = tesselation.sinusoidal(100 * u.deg**2)

Gallery
~~~~~~~
.. plot::
    :include-source: False

    from astropy import units as u
    from dorado.scheduling import tesselation
    from matplotlib import pyplot as plt
    import ligo.skymap.plot

    areas = np.asarray([1000, 500, 100, 50]) * u.deg**2
    methods = [tesselation.geodesic,
               tesselation.golden_angle_spiral,
               tesselation.sinusoidal]

    fig = plt.figure(figsize=(8, 6))
    gridspecs = fig.add_gridspec(len(methods) + 1, len(areas) + 1,
                                left=0, right=1, bottom=0, top=1,
                                 height_ratios=(1, *[10] * len(methods)),
                                 width_ratios=(1, *[10] * len(areas)))

    for i, method in enumerate(methods):
        ax = fig.add_subplot(gridspecs[i + 1, 0], frameon=False)
        ax.text(0.5, 0.5,
                method.__name__, rotation=90,
                ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    for j, area in enumerate(areas):
        ax = fig.add_subplot(gridspecs[0, j + 1], frameon=False)
        ax.text(0.5, 0.5,
                area.to_string(format='latex'),
                ha='center', va='center',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    for i, method in enumerate(methods):
        for j, area in enumerate(areas):
            ax = fig.add_subplot(gridspecs[i + 1, j + 1],
                                 projection='astro globe',
                                 center='0d 25d')
            for key in ['ra', 'dec']:
                ax.coords[key].set_ticklabel_visible(False)
                ax.coords[key].set_ticks_visible(False)
            ax.plot_coord(method(area), '.')
            ax.grid()
    # for gridspec_row, method in zip(gridspecs[1:], methods):
    #     ax = fig.add_subplot(gridspec_row[0], frameon=False)
    #     ax.text(0.5, 0.5, method.__name__, transform=ax.transAxes, rotation=90)

    #     for gridspec, area in zip(gridspec_row[1:], areas):
    #         ax = fig.add_subplot(gridspec,
    #                              projection='astro globe', center='0d 25d')
    #         for key in ['ra', 'dec']:
    #             ax.coords[key].set_ticklabel_visible(False)
    #             ax.coords[key].set_ticks_visible(False)
    #         ax.plot_coord(method(area), '.')
    #         ax.grid()
"""
from ._geodesic import geodesic
from ._spiral import golden_angle_spiral
from ._sinusoidal import sinusoidal

__all__ = ('geodesic', 'golden_angle_spiral', 'sinusoidal')
