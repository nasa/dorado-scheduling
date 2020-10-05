from astropy_healpix import HEALPix
from astropy.coordinates import ICRS


hpx = HEALPix(nside=32, frame=ICRS())
"""Base HEALpix resolution for all calculations."""
