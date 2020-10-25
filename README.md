# Dorado observation planning and scheduling simulations

Dorado is a proposed space mission for ultraviolet follow-up of gravitational
wave events. This repository contains a simple target of opportunity
observation planner for Dorado.

![Example Dorado observing plan](examples/6-5.gif)

## Features

*   **Global**: jointly and globally solves the problems of tiling (the set of
    telescope boresight orientations and roll angles) and the scheduling (which
    tile is observed at what time), rather than solving each sub-problem one at
    a time
*   **Optimal**: generally solves all the way to optimality, rather than
    finding merely a "good enough" solution
*   **Fast**: solve an entire orbit in about 5 minutes
*   **General**: does not depend on heuristics of any kind
*   **Flexible**: problem is formulated in the versatile framework of
    [mixed integer programming]

## Dependencies

*   [Astroplan] for calculating the field of regard
*   [HEALPix], [Healpy], and [astropy-healpix] for observation footprints
*   [Skyfield] for orbit propagation
*   [python-mip] and [Gurobi] for constrained optimization

## Problem formulation

Given a gravitational-wave HEALPix probability sky map, this Python package
finds an optimal sequence of Dorado observations to maximize the probability of
observing the (unknown) location of the gravitational-wave event, within one
orbit.

The problem is formulated as a mixed integer programming problem with the
following arrays of binary decision variables:

*   `schedule` (`npix` × `nrolls` × `ntimes - ntimes_per_exposure + 1`): 1 if
    an observation of the field that is centered on the given HEALPix pixel, at
    a given roll angle, is begun on a given time step; or 0 otherwise
*   `pix` (`npix`): 1 if the given HEALPix pixel is observed, or 0
    otherwise

The problem has the following parameters:

*   `nexp` (scalar, integer): the maximum number of exposures
*   `prob` (`npix`, float): the probability sky map
*   `regard` (`npix` × `ntimes`, binary): 1 if the field centered on
    the given HEALPix pixel is within the field of regard at the given time, 0
    otherwise

The objective function is the sum over all of the pixels in the probability sky
map for which the corresponding entry in `pix` is 1.

The constraints are:
*   At most one observation is allowed at a time.
*   At most `nexp` observations are allowed in total.
*   A given pixel is observed if any field that contains it within its
    footprint is observed.
*   A field may be observed only if it is within the field of regard.

## Usage

### To install

This Python project uses [Poetry] for packaging, dependency, and environment
management.

1.  First, [install Poetry]:

        $ curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

2.  Within the directory containing this README file, run the following command
    to create a Python virtual environment and install this package into it:

        $ poetry install

### To activate the virtual environment

Follow these steps to activate the environment each time you start a new shell
session.

3.  Activate the Poetry virtual environment:

        $ poetry shell

4.  It is recommended that you use the commercial [Gurobi] solver rather than
    the open-source Cbc solver that is included with python-mip; Cbc is much
    slower and may fail to solve to optimality. If you have a Gurobi license,
    then activate Gurobi in this shell by setting the `GUROBI_HOME` environment
    variable. For example, on macOS, this will be:

        $ export GUROBI_HOME=/Library/gurobi901/mac64

### To prepare the base problem

Generate a base problem once that you will reuse for every sky map and every
orbit.

5.  Generate the base problem by running the following command:

        $ dorado-scheduling-prepare model.lp.zst

This will take 5-20 minutes and will generate a file that is about 200 MB in
size.

### To generate an observing plan

6.  Generate an observing plan for the included example sky map:

        $ dorado-scheduling model.lp.zst examples/6.fits --nexp 5 -o 6-5.ecsv

    This will take 5-10 minutes and will use about 16 GB of memory at peak.

7.  Print out the observing plan:

        $ cat 6-5.ecsv 
        # %ECSV 0.9
        # ---
        # datatype:
        # - {name: time, datatype: string}
        # - {name: center.ra, unit: deg, datatype: float64}
        # - {name: center.dec, unit: deg, datatype: float64}
        # - {name: roll, unit: deg, datatype: float64}
        # meta:
        #   __serialized_columns__:
        #     center:
        #       __class__: astropy.coordinates.sky_coordinate.SkyCoord
        #       dec: !astropy.table.SerializedColumn
        #         __class__: astropy.coordinates.angles.Latitude
        #         unit: &id001 !astropy.units.Unit {unit: deg}
        #         value: !astropy.table.SerializedColumn {name: center.dec}
        #       frame: icrs
        #       ra: !astropy.table.SerializedColumn
        #         __class__: astropy.coordinates.angles.Longitude
        #         unit: *id001
        #         value: !astropy.table.SerializedColumn {name: center.ra}
        #         wrap_angle: !astropy.coordinates.Angle
        #           unit: *id001
        #           value: 360.0
        #       representation_type: spherical
        #     time:
        #       __class__: astropy.time.core.Time
        #       format: isot
        #       in_subfmt: '*'
        #       out_subfmt: '*'
        #       precision: 3
        #       scale: utc
        #       value: !astropy.table.SerializedColumn {name: time}
        #   cmdline: dorado-scheduling model.lp.zst examples/6.fits -o 6-5.ecsv --nexp 5
        #   prob: 0.8881919618423365
        #   real: 124.77953617399999
        #   status: OPTIMAL
        #   sys: 21.667351000000004
        #   user: 144.99249299999997
        # schema: astropy-2.0
        time center.ra center.dec roll
        2012-05-02T19:09:32.699 51.74999999999999 -60.434438844952275 80.0
        2012-05-02T19:20:32.699 65.25 -60.434438844952275 0.0
        2012-05-02T19:30:32.699 79.28571428571429 -58.91977535280316 80.0
        2012-05-02T19:43:32.699 91.95652173913044 -55.87335043525197 60.0
        2012-05-02T19:53:32.699 133.59375 7.180755781458282 20.0

7.  To generate an animated visualization for this observing plan, run the
    following command:

        $ dorado-scheduling-animate exaples/6.fits 6-5.ecsv 6-5.gif

    This will take 2-5 minutes to run.

[mixed integer programming]: https://en.wikipedia.org/wiki/Integer_programming
[Astroplan]: https://github.com/astropy/astroplan
[HEALPix]: https://healpix.jpl.nasa.gov
[astropy-healpix]: https://github.com/astropy/astropy-healpix
[Healpy]: https://github.com/healpy/healpy
[Skyfield]: https://rhodesmill.org/skyfield/
[python-mip]: https://python-mip.com
[Poetry]: https://python-poetry.org
[install Poetry]: https://python-poetry.org/docs/#installation
[Gurobi]: https://www.gurobi.com