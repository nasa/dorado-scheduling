#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Create a tesselation."""
import logging
from astropy.coordinates import SkyCoord
import numpy as np

from ligo.skymap.tool import ArgumentParser, FileType

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('fovsize', default=3.3,
                   type=float,
                   help='in degrees.')
    p.add_argument('scale', default=0.8,
                   type=float,
                   help='scale for overlap.')
    p.add_argument('output', metavar='tess.dat', type=FileType('wb'),
                   help='Output filename')
    return p


def tesselation_spiral(FOV_size, scale=0.80):
    FOV = FOV_size*FOV_size*scale

    area_of_sphere = 4*np.pi*(180/np.pi)**2
    n = int(np.ceil(area_of_sphere/FOV))

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    ra, dec = hp.pixelfunc.vec2ang(points, lonlat=True)
    return ra, dec


def main(args=None):
    args = parser().parse_args(args)

    log.info('creating tesselation')
    ras, decs = tesselation_spiral(args.fovsize, scale=args.scale)

    fid = open(args.output.name, 'w')
    for ii in range(len(ras)):
        fid.write('%d %.5f %.5f\n' % (ii, ras[ii], decs[ii]))
    fid.close()


if __name__ == '__main__':
    main()
