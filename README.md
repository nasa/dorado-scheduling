# Dorado observation planning and scheduling simulations

[![Build Status](https://github.com/nasa/dorado-scheduling/actions/workflows/build-and-test.yml/badge.svg)](https://github.com/nasa/dorado-scheduling/actions)
[![Documentation Status](https://readthedocs.org/projects/dorado-scheduling/badge/?version=latest)](https://dorado-scheduling.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://img.shields.io/codecov/c/github/nasa/dorado-scheduling)](https://app.codecov.io/gh/nasa/dorado-scheduling)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/dorado-scheduling)](https://pypi.org/project/dorado-scheduling/)

Dorado is a proposed space mission for ultraviolet follow-up of gravitational
wave events. This repository contains a simple target of opportunity
observation planner for Dorado.

**To get started, see the [quick start instructions] in the [manual].**

![Example Dorado observing plan](examples/6.gif)

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

*   [Astropy]
*   [Astroplan] for calculating the field of regard
*   [HEALPix], [cdshealpix], and [astropy-healpix] for observation footprints
*   [sgp4] for orbit propagation
*   [CPLEX] (via [docplex] Python interface) for constrained optimization

[quick start instructions]: https://dorado-scheduling.readthedocs.io/en/latest/quickstart.html
[manual]: https://dorado-scheduling.readthedocs.io/
[mixed integer programming]: https://en.wikipedia.org/wiki/Integer_programming
[Astropy]: https://www.astropy.org
[Astroplan]: https://github.com/astropy/astroplan
[HEALPix]: https://healpix.jpl.nasa.gov
[astropy-healpix]: https://github.com/astropy/astropy-healpix
[cdshealpix]: https://github.com/cds-astro/cds-healpix-python
[sgp4]: https://pypi.org/project/sgp4/
[CPLEX]: https://www.ibm.com/products/ilog-cplex-optimization-studio
