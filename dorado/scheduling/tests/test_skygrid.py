#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
from functools import partial

from astropy import units as u
import pytest

from .. import skygrid


geodesic_methods = [
    partial(skygrid.geodesic, base=base, class_=class_)
    for base in ['icosahedron', 'octahedron', 'tetrahedron']
    for class_ in ['I', 'II', 'III']]


@pytest.mark.parametrize('method', [
    *geodesic_methods,
    skygrid.golden_angle_spiral,
    skygrid.healpix,
    skygrid.sinusoidal
])
@pytest.mark.parametrize('area', [10, 100, 1000] * u.deg**2)
def test_skygrid(method, area):
    coords = method(area)
    assert len(coords) >= (u.spat / area)


def test_invalid_geodesic_polyhedron():
    with pytest.raises(ValueError):
        # No such thing as a class IV geodesic polyhedron
        skygrid.geodesic(10 * u.deg**2, class_='IV')
