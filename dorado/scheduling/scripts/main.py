"""Command line interface."""
import logging
import tempfile

from astropy_healpix import nside_to_level
from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from ligo.skymap.io import read_sky_map
from ligo.skymap.bayestar import rasterize
from ligo.skymap.tool import ArgumentParser, FileType
import mip
import numpy as np
from scipy.signal import convolve
from tqdm import tqdm
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
    p.add_argument('-n', '--nexp', type=int, default=orbit.exposures_per_orbit,
                   help='Number of exposures')
    p.add_argument('--max-seconds', type=int, default=300,
                   help='Time limit for solver')
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

    times = np.arange(orbit.time_steps) * orbit.exposure_time / orbit.time_steps_per_exposure \
        + start_time

    log.info('reading initial model')
    m = mip.Model()
    with tempfile.NamedTemporaryFile(suffix='.lp') as uncompressed:
        ZstdDecompressor().copy_stream(args.model, uncompressed)
        m.read(uncompressed.name)

    log.info('reconstructing tensor variables')
    schedule = np.asarray(
        [[[m.var_by_name(f's_{i}_{j}_{k}')
           for k in range(orbit.time_steps - orbit.time_steps_per_exposure + 1)]
          for j in range(len(skygrid.rolls))]
         for i in tqdm(range(len(skygrid.centers)))]).view(mip.LinExprTensor)
    pixel_observed = np.asarray([
        m.var_by_name(f'p_{i}')
        for i in tqdm(range(skygrid.healpix.npix))]).view(mip.LinExprTensor)

    m.constr_by_name('nexp').rhs = args.nexp

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(~get_field_of_regard(times),
        np.ones(orbit.time_steps_per_exposure)[:, np.newaxis],
        mode='valid', method='direct'))
    m += mip.xsum(schedule[j, :, i].ravel()) <= 0

    log.info('adding objective')
    m.objective = mip.maximize(mip.xsum(prob * pixel_observed))

    log.info('solving')
    m.optimize(max_seconds=args.max_seconds)

    print('Fields observed:')
    if m.status in {mip.OptimizationStatus.FEASIBLE,
                    mip.OptimizationStatus.OPTIMAL}:
        schedule_flags = schedule.astype(float).astype(bool)
        objective_value = m.objective_value
    else:
        schedule_flags = np.zeros(schedule.shape, dtype=bool)
        objective_value = 0.0

    ipix, iroll, itime = np.nonzero(schedule_flags)
    result = Table(
        {
            'time': times[itime],
            'center': skygrid.centers[ipix],
            'roll': skygrid.rolls[iroll]
        }, meta={
            'prob': objective_value,
            'status': m.status.name
        }
    )
    result.sort('time')
    print(result)

    log.info('done')


if __name__ == '__main__':
    main()
