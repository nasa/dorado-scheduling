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
import astropy.units as u
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


def tesselation_spiral_packing(FOV_size, scale=0.80):
    """
    Spiral-based spherical packing scheme for tiling
    Used by GRANDMA follow-up during O3 (2004.04277)

    :param FOV_size: FOV in degrees for packing
    :param scale: scaling term for tile overlaps

    :return: SkyCoord
    """

    FOV = FOV_size*FOV_size*scale

    area_of_sphere = 4*np.pi*(180/np.pi)**2
    n = int(np.ceil(area_of_sphere/FOV))

    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1 - 1.0 / n, 1.0 / n - 1, n)
    radius = np.sqrt(1 - z * z)

    coords = SkyCoord(radius, theta * u.rad, z,
                      representation_type='cylindrical')
    coords.representation_type = 'unitspherical'
    return coords


def tesselation_phi_theta_packing(FOV_size, scale=0.97):
    """
    Phi/Theta spherical packing scheme for tiling
    Used by GROWTH DECam follow-up during O3 (1906.00806)

    :param FOV_size: FOV in degrees for packing
    :param scale: scaling term for tile overlaps

    :return: SkyCoord
    """

    sphere_radius = 1.0
    circle_radius = np.deg2rad(FOV_size/2.0) * scale
    vertical_count = int((np.pi*sphere_radius)/(2*circle_radius))

    phis = []
    thetas = []

    phi = -0.5*np.pi
    phi_step = np.pi/vertical_count
    while phi < 0.5*np.pi:
        horizontal_count = int((2*np.pi*np.cos(phi)*sphere_radius) /
                               (2*circle_radius))
        if horizontal_count == 0:
            horizontal_count = 1
        theta = 0
        theta_step = 2*np.pi/horizontal_count
        while theta < 2*np.pi-1e-8:
            phis.append(phi)
            thetas.append(theta)
            theta += theta_step
        phi += phi_step
    dec = np.array(np.rad2deg(phis))
    ra = np.array(np.rad2deg(thetas))

    coords = SkyCoord(ra * u.deg, dec*u.deg)
    coords.representation_type = 'unitspherical'
    return coords


def main(args=None):
    args = parser().parse_args(args)

    log.info('creating tesselation')
    coords = tesselation_phi_theta_packing(args.fovsize, scale=args.scale)

    fid = open(args.output.name, 'w')
    for ii, coord in enumerate(coords):
        fid.write('%d %.5f %.5f\n' % (ii, coord.ra.deg, coord.dec.deg))
    fid.close()


if __name__ == '__main__':
    main()
