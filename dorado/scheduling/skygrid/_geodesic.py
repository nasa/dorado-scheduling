#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from math import gcd

from anti_lib_progs.geodesic import get_poly, grid_to_points, make_grid
from astropy.coordinates import SkyCoord
from astropy import units as u
import numpy as np


def triangulation_number(b, c):
    return b * b + b * c + c * c


def solve_number_of_vertices(n, base, class_):
    base_count = {
        'icosahedron': 10,
        'octahedron': 4,
        'tetrahedron': 2
    }[base]

    if class_ == 'I':
        b = int(np.ceil(np.sqrt((n - 2) / base_count)))
        c = 0
        t = b * b
    elif class_ == 'II':
        b = c = int(np.ceil(np.sqrt((n - 2) / (base_count * 3))))
        t = 3 * b * b
    elif class_ == 'III':
        # FIXME: This is a brute-force search.
        # This could be solved easily by Gurobi as a non-convex MIQCQP problem,
        # or by a custom dynamic programming solution.
        b_max = int(np.ceil(np.sqrt((n - 2) / base_count)))
        t, c, b = min((triangulation_number(b, c), c, b)
                      for b in range(b_max + 1)
                      for c in range(b_max + 1)
                      if triangulation_number(b, c) * base_count + 2 >= n)
    else:
        raise ValueError('Unknown breakdown class')

    return base_count * t + 2, t, b, c


def geodesic(area, base='icosahedron', class_='I'):
    """Generate a geodesic polyhedron with the fewest vertices >= `n`.

    Parameters
    ----------
    area : :class:`astropy.units.Quantity`
        The average area per tile in any Astropy solid angle units:
        for example, :samp:`10 * astropy.units.deg**2` or
        :samp:`0.1 * astropy.units.steradian`.
    base : {``'icosahedron'``,  ``'octahedron'``, ``'tetrahedron'``}
        The base polyhedron of the tesselation.
    class_ : {``'I'``, ``'II'``, ``'III'``}
        The class of the geodesic polyhedron, which constrains the allowed
        values of the number of points. Class III permits the most freedom.

    Returns
    -------
    coords : :class:`astropy.coordinates.SkyCoord`
        The coordinates of the vertices of the geodesic polyhedron.

    See also
    --------
    <https://en.wikipedia.org/wiki/Geodesic_polyhedron>

    Example
    -------

    .. plot::
        :context: reset

        from astropy import units as u
        from matplotlib import pyplot as plt
        import ligo.skymap.plot
        import numpy as np

        from dorado.scheduling import skygrid

        n_vertices_target = 1024
        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target)
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class I')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    .. plot::
        :context: close-figs

        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target,
                                    class_='II')
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class II')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    .. plot::
        :context: close-figs

        vertices = skygrid.geodesic(4 * np.pi * u.sr / n_vertices_target,
                                        class_='III')
        n_vertices = len(vertices)

        ax = plt.axes(projection='astro globe', center='0d 25d')
        plt.suptitle('Class III')
        ax.set_title(f'{n_vertices} vertices (goal was {n_vertices_target})')
        ax.plot_coord(vertices, '.')
        ax.grid()

    """
    n = int(np.ceil(1 / area.to_value(u.spat)))

    # Adapted from
    # https://github.com/antiprism/antiprism_python/blob/master/anti_lib_progs/geodesic.py
    verts = []
    edges = {}
    faces = []
    get_poly(base[0], verts, edges, faces)

    n, t, b, c = solve_number_of_vertices(n, base, class_)
    divisor = gcd(b, c)
    t //= divisor
    b //= divisor
    c //= divisor

    grid = {}
    grid = make_grid(t, b, c)

    points = verts
    for face in faces:
        points[len(points):len(points)] = grid_to_points(
            grid, t, False, [verts[face[i]] for i in range(3)], face)

    assert len(points) == n
    coords = SkyCoord(*zip(*(point.v for point in points)),
                      representation_type='cartesian')
    coords = SkyCoord(coords, representation_type='unitspherical')
    return coords
