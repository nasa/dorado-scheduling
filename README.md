# Dorado observation planning and scheduling simulations

Dorado is a proposed space mission for ultraviolet follow-up of gravitational
wave events. This repository contains a simple target of opportunity
observation planner for Dorado.

* Field of regard is calculated with [Astroplan]
* Field of view footprints using [HEALPix] ([Healpy] + [astropy-healpix])
* Orbit propagation with [Skyfield]
* Constrained optimimzation with [python-mip]

[Astroplan]: https://github.com/astropy/astroplan
[HEALPix]: https://healpix.jpl.nasa.gov
[astropy-healpix]: https://github.com/astropy/astropy-healpix
[Healpy]: https://github.com/healpy/healpy
[Skyfield]: https://rhodesmill.org/skyfield/
[python-mip]: https://python-mip.com
