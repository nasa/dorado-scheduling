"""Command line interface."""
import logging
import tempfile

from astropy_healpix import nside_to_level
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from ligo.skymap.tool import ArgumentParser, FileType
import mip
import numpy as np
from zstandard import ZstdDecompressor

from .. import orbit
from ..regard import get_field_of_regard
from .. import skygrid

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('model', metavar='MODEL.lp.zst', type=FileType('rb'),
                   help='Prepared model')
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('-n', '--nexp', type=int, default=10,
                   help='Number of exposures')
    return p


def main(args=None):
    args = parser().parse_args(args)

    log.info('reading sky map')

    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(fits.getval(args.skymap, 'DATE-OBS', ext=1))
    skymap = read_sky_map(args.skymap, moc=True)
    skymap['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap, nside_to_level(skygrid.healpix.nside))['PROB']
    if skygrid.healpix.order == 'ring':
        prob = prob[skygrid.healpix.ring_to_nested(np.arange(len(prob)))]

    log.info('reading initial model')
    m = mip.Model()
    with tempfile.NamedTemporaryFile(suffix='.lp') as uncompressed:
        ZstdDecompressor().copy_stream(args.model, uncompressed)
        m.read(uncompressed.name)

    log.info('reconstructing tensor variables')
    field_observed = np.asarray(
        [[m.var_by_name(f'field_observed_{i}_{j}')
          for j in range(len(skygrid.rolls))]
         for i in range(len(skygrid.centers))]).view(mip.LinExprTensor)
    pixel_observed = np.asarray([
        m.var_by_name(f'pixel_observed_{i}')
        for i in range(skygrid.healpix.npix)]).view(mip.LinExprTensor)

    # log.info('adding variable: whether a given field is observed in a given time slot')
    # schedule = m.add_var_tensor(
    #     (orbit.exposures_per_orbit, len(skygrid.centers), len(skygrid.rolls)),
    #     'schedule', var_type=mip.BINARY)

    # log.info('adding constraint: only one field observed at a time')
    # for slice in schedule:
    #     m += mip.xsum(slice.ravel()) <= 1

    # log.info('adding constraint: a field is observed if it is observed at any time')
    # for i in range(len(skygrid.centers)):
    #     for j in range(len(skygrid.rolls)):
    #         m += mip.xsum(schedule[:, i, j]) >= field_observed[i, j]

    log.info('adding constraint: number of exposures')
    m += m.var_by_name('num_fields') <= args.nexp

    log.info('adding objective')
    m.objective = mip.maximize(mip.xsum(prob * pixel_observed))

    status = m.optimize(max_seconds=300)
    print(status)

    print('Fields observed:')
    ipix, iroll = np.nonzero(field_observed.astype(float))
    print(Table({'center': skygrid.centers[ipix],
                 'roll': skygrid.rolls[iroll]}))

    log.info('done')


if __name__ == '__main__':
    main()
