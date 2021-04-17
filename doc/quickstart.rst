Quick Start
===========

To install
----------

To install with `Pip`_:

1.  Run the following command::

        $ pip install git+https://github.com/dorado-science/dorado-scheduling

To set up the CPLEX optimization engine
---------------------------------------

2.  Set up the CPLEX optimization engine by following the
    `docplex instructions`_. If you have `installed CPLEX locally`_, then all you
    have to do is determine the path to the CPLEX Python bindings and add them
    to your :envvar:`PYTHONPATH`. For example, on macOS, this might be::

        $ export PYTHONPATH=/Applications/CPLEX_Studio1210/cplex/python/3.7/x86-64_osx

To generate an observing plan
-----------------------------

3.  Generate an observing plan for the included example sky map::

        $ dorado-scheduling examples/6.fits -o examples/6.ecsv

    This will take 3 minutes and will use about 10 GB of memory at peak.

4.  Print out the observing plan::

        $ cat examples/6.ecsv 
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
        #   cmdline: dorado-scheduling examples/6.fits -o examples/6.ecsv
        #   prob: 0.8238346278346581
        #   status: OPTIMAL
        # schema: astropy-2.0
        time center.ra center.dec roll
        2012-05-02T18:29:32.699 54.47368421052631 -61.94383702315671 80.0
        2012-05-02T18:44:32.699 69.75 -60.434438844952275 50.0
        2012-05-02T19:32:32.699 147.65625 -7.180755781458282 70.0
        2012-05-02T19:42:32.699 115.31249999999999 18.20995686428301 20.0
        2012-05-02T19:56:32.699 133.59375 7.180755781458282 20.0

5.  To generate an animated visualization for this observing plan, run the
    following command::

        $ dorado-scheduling-animate examples/6.fits examples/6.ecsv examples/6.gif

    This will take about 2-5 minutes to run.

Determining if a given sky position is contained within an observing plan
-------------------------------------------------------------------------

The following example illustrates how to use HEALPix to determine if a given
sky position is contained in any of the fields in an observing plan::

    >>> from astropy.coordinates import ICRS, SkyCoord
    >>> from astropy.table import QTable
    >>> from astropy_healpix import HEALPix
    >>> from astropy import units as u
    >>> from dorado.scheduling import FOV
    >>> healpix = HEALPix(nside=32, frame=ICRS())
    >>> target = SkyCoord(66.91436579*u.deg, -61.98378895*u.deg)
    >>> target_pixel = healpix.skycoord_to_healpix(target)
    >>> schedule = QTable.read('examples/6.ecsv')
    >>> fov = FOV.from_rectangle(7.1 * u.deg)
    >>> footprints = [fov.footprint_healpix(healpix, row['center'], row['roll'])
    ...               for row in schedule]
    >>> schedule['found'] = [target_pixel in footprint for footprint in footprints]
    >>> schedule
    <QTable length=8>
              time          exptime ...   roll  found
                              min   ...   deg        
             object         float64 ... float64  bool
    ----------------------- ------- ... ------- -----
    2012-05-02T18:28:32.699    10.0 ...    60.0 False
    2012-05-02T18:38:32.699    10.0 ...    40.0 False
    2012-05-02T18:48:32.699    10.0 ...    80.0 False
    2012-05-02T18:58:32.699    10.0 ...    50.0  True
    2012-05-02T19:23:32.699    10.0 ...    80.0 False
    2012-05-02T19:33:32.699    10.0 ...    20.0 False
    2012-05-02T19:43:32.699    10.0 ...    20.0 False
    2012-05-02T19:57:32.699    10.0 ...     0.0 False

.. _`Pip`: https://pip.pypa.io
.. _`mixed integer programming`: https://en.wikipedia.org/wiki/Integer_programming
.. _`Astropy`: https://www.astropy.org
.. _`Astroplan`: https://github.com/astropy/astroplan
.. _`HEALPix`: https://healpix.jpl.nasa.gov
.. _`astropy-healpix`: https://github.com/astropy/astropy-healpix
.. _`Healpy`: https://github.com/healpy/healpy
.. _`Skyfield`: https://rhodesmill.org/skyfield/
.. _`install Poetry`: https://python-poetry.org/docs/#installation
.. _`CPLEX`: https://www.ibm.com/products/ilog-cplex-optimization-studio
.. _`docplex`: https://ibmdecisionoptimization.github.io/docplex-doc/
.. _`docplex instructions`: https://ibmdecisionoptimization.github.io/docplex-doc/mp/getting_started.html
.. _`installed CPLEX locally`: https://ibmdecisionoptimization.github.io/docplex-doc/mp/getting_started.html#using-ibm-ilog-cplex-optimization-studio-on-your-computer
