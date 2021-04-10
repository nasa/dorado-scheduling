#
# Copyright © 2020 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Command line interface."""
import logging

from ligo.skymap.tool import ArgumentParser, FileType

log = logging.getLogger(__name__)


def parser():
    p = ArgumentParser()
    p.add_argument('skymap', metavar='FILE.fits[.gz]',
                   type=FileType('rb'), help='Input sky map')
    p.add_argument('tiles', metavar='FILE.dat',
                   type=FileType('rb'), help='tiling file')
    p.add_argument('-n', '--nexp', type=int, help='Number of exposures')
    p.add_argument('-s', '--start_time', type=str,
                   default='2020-01-01T00:00:00')
    p.add_argument('--output', '-o', metavar='OUTPUT.ecsv',
                   type=FileType('w'), default='-',
                   help='output filename')
    p.add_argument('-j', '--jobs', type=int, default=1, const=None, nargs='?',
                   help='Number of threads')
    p.add_argument('-c', '--config', help='config file')

    return p


def main(args=None):
    args = parser().parse_args(args)

    # Late imports
    import os
    # import shlex
    import sys

    from astropy_healpix import nside_to_level
    from astropy.time import Time
    from astropy.table import Table, QTable
    from astropy import units as u
    import configparser
    from docplex.mp.model import Model
    from ligo.skymap.io import read_sky_map
    from ligo.skymap.bayestar import rasterize
    from ligo.skymap.util import Stopwatch
    import numpy as np
    from scipy.signal import convolve
    from tqdm import tqdm

    from ..models import TilingModel

    tiles = QTable.read(args.tiles, format='ascii.ecsv')

    if args.config is not None:
        config = configparser.ConfigParser()
        config.read(args.config)
        satfile = config["survey"]["satfile"]
        exposure_time = float(config["survey"]["exposure_time"]) * u.minute
        steps_per_exposure =\
            int(config["survey"]["time_steps_per_exposure"])
        field_of_view = float(config["survey"]["field_of_view"]) * u.deg
        number_of_orbits = int(config["survey"]["number_of_orbits"])
        tiling_model = TilingModel(satfile=satfile,
                                   exposure_time=exposure_time,
                                   time_steps_per_exposure=steps_per_exposure,
                                   field_of_view=field_of_view,
                                   number_of_orbits=number_of_orbits,
                                   centers=tiles["center"])
    else:
        tiling_model = TilingModel(centers=tiles["center"])

    log.info('reading sky map')
    # Read multi-order sky map and rasterize to working resolution
    start_time = Time(args.start_time, format='isot')
    skymap = read_sky_map(args.skymap, moc=True)['UNIQ', 'PROBDENSITY']
    prob = rasterize(skymap,
                     nside_to_level(tiling_model.healpix.nside))['PROB']
    if tiling_model.healpix.order == 'ring':
        prob = prob[tiling_model.healpix.ring_to_nested(np.arange(len(prob)))]

    times = np.arange(tiling_model.time_steps) *\
        tiling_model.time_step_duration + start_time

    log.info('generating model')
    m = Model()
    m.set_time_limit(300)
    if args.jobs is not None:
        m.context.cplex_parameters.threads = args.jobs

    log.info('adding variable: observing schedule')
    shape = (len(tiling_model.centers),
             tiling_model.time_steps -
             tiling_model.time_steps_per_exposure + 1)
    schedule = np.reshape(m.binary_var_list(np.prod(shape)), shape)

    log.info('adding variable: whether a given pixel is observed')
    pixel_observed = np.asarray(m.binary_var_list(tiling_model.healpix.npix))

    log.info('adding variable: whether a given field is used')
    field_used = np.asarray(m.binary_var_list(shape[0]))

    log.info('adding variable: whether a given time step is used')
    time_used = np.asarray(m.binary_var_list(shape[1]))

    if args.nexp is not None:
        log.info('adding constraint: number of exposures')
        m.add_constraint_(m.sum(time_used) <= args.nexp)

    log.info('adding constraint: only observe one field at a time')
    m.add_constraints_(
        m.sum(schedule[..., i].ravel()) <= 1 for i in tqdm(range(shape[1]))
    )
    m.add_equivalences(
        time_used,
        [m.sum(schedule[..., i].ravel()) >= 1 for i in tqdm(range(shape[1]))]
    )
    m.add_constraints_(
        m.sum(time_used[i:i+tiling_model.time_steps_per_exposure]) <= 1
        for i in tqdm(range(schedule.shape[-1]))
    )

    log.info('adding constraint: a pixel is observed if it is in any field')
    m.add_constraints_(
        m.sum(lhs) >= rhs
        for lhs, rhs in zip(
            tqdm(schedule.reshape(field_used.size, -1)),
            field_used.ravel()
        )
    )

    indices = [[] for _ in range(tiling_model.healpix.npix)]
    with tqdm(total=len(tiling_model.centers)) as progress:
        for i, center in enumerate(tiling_model.centers):
            grid_ij = tiling_model.get_footprint_healpix(center)
            for k in grid_ij:
                indices[k].append(i)
            progress.update()
    m.add_constraints_(
        m.sum(field_used[lhs_index] for lhs_index in lhs_indices) >= rhs
        for lhs_indices, rhs in zip(tqdm(indices), pixel_observed)
    )

    log.info('adding constraint: field of regard')
    i, j = np.nonzero(
        convolve(
            ~tiling_model.get_field_of_regard(times, jobs=args.jobs),
            np.ones(tiling_model.time_steps_per_exposure)[:, np.newaxis],
            mode='valid', method='direct'))
    m.add_constraint_(m.sum(schedule[j, i].ravel()) <= 0)

    log.info('adding objective')
    m.maximize(m.scal_prod(pixel_observed, prob))

    log.info('solving')
    stopwatch = Stopwatch()
    stopwatch.start()
    solution = m.solve(log_output=True)
    stopwatch.stop()

    log.info('extracting results')
    if solution is None:
        schedule_flags = np.zeros(schedule.shape, dtype=bool)
        objective_value = 0.0
    else:
        schedule_flags = np.asarray(
            solution.get_values(schedule.ravel()), dtype=bool
        ).reshape(
            schedule.shape
        )
        objective_value = m.objective_value

    from cplex.callbacks import LazyConstraintCallback
    from docplex.mp.callbacks.cb_mixin import ConstraintCallbackMixin

    def neighbors(node, sol, x, Edges):
        """Get the neighbors of NODE in the current tour in SOL."""
        return \
            [e[1] for e in Edges if e[0] == node and
             sol.get_value(x[e]) > 0.5] + \
            [e[0] for e in Edges if e[1] == node and sol.get_value(x[e]) > 0.5]

    # Lazy constraint callback to separate subtour elimination constraints.
    class DOLazyCallback(ConstraintCallbackMixin, LazyConstraintCallback):
        def __init__(self, env):
            LazyConstraintCallback.__init__(self, env)
            ConstraintCallbackMixin.__init__(self)

        def __call__(self):
            # Fetch variable values into a solution object
            sol = self.make_solution_from_vars(self.x.values())
            visited = set()
            for i in self.Cities:
                if i in visited:
                    continue
                # Find the (sub)tour that includes city i
                start = i
                node = i
                subtour = [-1] * self.n
                size = 0
                # Loop until we get back to start
                nodes = list()
                while node != start or size == 0:
                    visited.add(node)
                    nodes.append(node)
                    # Pick the neighbor that we did not yet visit
                    # on this (sub)tour
                    succ = None
                    for j in neighbors(node, sol, self.x, self.Edges):
                        if j == start or j not in visited:
                            succ = j
                            break
                    # Move to the next neigbor
                    subtour[node] = succ
                    node = succ
                    size += 1
                # If the tour does not touch every node then it is a subtour
                # and needs to be eliminated
                if size < self.n:
                    # Create a constraint that states that from
                    # the variables in the subtour not all can be 1.
                    tour = 0
                    for j, k in enumerate(subtour):
                        if k >= 0:
                            tour += self.x[(min(j, k), max(j, k))]
                    ct = tour <= size - 1
                    unsats = self.get_cpx_unsatisfied_cts([ct],
                                                          sol,
                                                          tolerance=1e-6)
                    for ct, cpx_lhs, sense, cpx_rhs in unsats:
                        self.add(cpx_lhs, sense, cpx_rhs)
                    # Stop separation, we separate only one subtour at a time.
                    break

    ipix, itime = np.nonzero(schedule_flags)
    cent = tiling_model.centers[ipix]
    n = len(cent)

    if n > 1:
        Edges = []
        dist = {}
        for ii in range(len(cent)):
            for jj in range(len(cent)):
                if ii >= jj:
                    continue
                cent1, cent2 = cent[ii], cent[jj]
                edge = (ii, jj)
                dist[edge] = int(cent1.separation(cent2).arcsecond)
                Edges.append(edge)

        Cities = range(n)
        m = Model(name='tsp')
        m.set_time_limit(60)

        x = m.binary_var_dict(Edges)
        m.minimize(m.sum(dist[e] * x[e] for kk, e in enumerate(Edges)))

        # Register a lazy constraint callback
        cb = m.register_callback(DOLazyCallback)

        # Each city is linked with two other cities
        for j in Cities:
            m.add_constraint(sum(x[e] for e in Edges if e[0] == j) +
                             sum(x[e] for e in Edges if e[1] == j) == 2)

        # Store references to variables in callback instance so that we can use
        # it for separation
        cb.n = n
        cb.Edges = Edges
        cb.Cities = Cities
        cb.x = x
        m.lazy_callback = cb

        # Solve the model.
        m.solve(log_output=True, TimeLimit=60)
        sol = m.solution

        if sol is not None:
            tour = list()
            start = Cities[0]
            node = start
            visited = set()
            while len(tour) == 0 or node is not start:
                tour.append(node)
                visited.add(node)
                for j in neighbors(node, sol, x, Edges):
                    if j == start or j not in visited:
                        neighbor = j
                        break
                node = neighbor
            print('Optimal tour: %s' % ' - '.join([str(j) for j in tour]))

            ipix, itime = np.nonzero(schedule_flags)
            ipix = ipix[tour]

    result = Table(
        {
            'time': times[itime],
            'center': tiling_model.centers[ipix],
        }, meta={
            # FIXME: use shlex.join(sys.argv) in Python >= 3.8
            'cmdline': ' '.join(sys.argv),
            'prob': objective_value,
            'status': m.solve_status.name,
            'real': stopwatch.real,
            'user': stopwatch.user,
            'sys': stopwatch.sys
        }
    )
    result.sort('time')
    result.write(args.output.name, format='ascii.ecsv')

    log.info('done')
    # Fast exit without garbage collection
    args.output.close()
    os._exit(os.EX_OK)


if __name__ == '__main__':
    main()
