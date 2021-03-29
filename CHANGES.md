# Changelog

## Version 0.1.1 (unreleased)

-   Add the tool ``dorado-scheduling-tile`` tool to generate all-sky survey
    grids.

-   Replace the South Atlantic Anomaly constraint with the more general
    AE-8/AP-8 Van Allen belt model that is applicable across a wider range of
    orbits, energies, and fluxes.

-   Switch the implementation of the orbit propagation from the high-level
    Skyfield interface to the low-level SGP4 interface. This fixes incorrect
    satellite positions due to incorrect Astropy coordinates frames (see
    [python-skyfield#577](https://github.com/skyfielders/python-skyfield/issues/577)).

## Version 0.1.0 (2021-02-02)

-   First public release.
