#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Create a tesselation."""
import astropy.units as u
from astropy.table import QTable
import numpy as np

from ligo.skymap.tool import ArgumentParser, FileType

from .. import tesselation

METHODS = {'geodesic': tesselation.geodesic,
           'golden-angle-spiral': tesselation.golden_angle_spiral,
           'sinusoidal': tesselation.sinusoidal}


def parser():
    p = ArgumentParser()
    p.add_argument('--area', default='50 deg2', type=u.Quantity,
                   help='Average area per tile')
    p.add_argument('--method', default='geodesic', help='Tiling algorithm',
                   choices=tuple(METHODS.keys()))
    p.add_argument('-o', '--output', metavar='OUTPUT.ecsv',
                   type=FileType('wb'), help='Output filename')
    return p


def main(args=None):
    args = parser().parse_args(args)

    method = METHODS[args.method]
    coords = method(args.area)
    table = QTable({'field_id': np.arange(len(coords)), 'center': coords})
    table.write(args.output, format='ascii.ecsv')


if __name__ == '__main__':
    main()
