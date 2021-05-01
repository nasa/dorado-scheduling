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
        # - {name: time, datatype: string, description: Start time of observation}
        # - {name: exptime, unit: min, datatype: float64, description: Exposure time}
        # - {name: location.x, unit: km, datatype: float64, description: Location of the spacecraft}
        # - {name: location.y, unit: km, datatype: float64, description: Location of the spacecraft}
        # - {name: location.z, unit: km, datatype: float64, description: Location of the spacecraft}
        # - {name: center.ra, unit: deg, datatype: float64}
        # - {name: center.dec, unit: deg, datatype: float64}
        # - {name: roll, unit: deg, datatype: float64, description: 'Roll angle of spacecraft, position angle of FOV'}
        # meta:
        #   __serialized_columns__:
        #     center:
        #       __class__: astropy.coordinates.sky_coordinate.SkyCoord
        #       __info__: {description: Pointing of the center of the spacecraft's FOV}
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
        #     location:
        #       __class__: astropy.coordinates.earth.EarthLocation
        #       __info__: {description: Location of the spacecraft}
        #       ellipsoid: WGS84
        #       x: !astropy.table.SerializedColumn
        #         __class__: astropy.units.quantity.Quantity
        #         __info__: {description: Location of the spacecraft}
        #         unit: &id002 !astropy.units.Unit {unit: km}
        #         value: !astropy.table.SerializedColumn {name: location.x}
        #       y: !astropy.table.SerializedColumn
        #         __class__: astropy.units.quantity.Quantity
        #         __info__: {description: Location of the spacecraft}
        #         unit: *id002
        #         value: !astropy.table.SerializedColumn {name: location.y}
        #       z: !astropy.table.SerializedColumn
        #         __class__: astropy.units.quantity.Quantity
        #         __info__: {description: Location of the spacecraft}
        #         unit: *id002
        #         value: !astropy.table.SerializedColumn {name: location.z}
        #     time:
        #       __class__: astropy.time.core.Time
        #       __info__: {description: Start time of observation}
        #       format: isot
        #       in_subfmt: '*'
        #       out_subfmt: '*'
        #       precision: 3
        #       scale: utc
        #       value: !astropy.table.SerializedColumn {name: time}
        #   cmdline: dorado-scheduling examples/6.fits -o examples/6.ecsv
        #   prob: 0.9582737898644651
        #   real: 82.14122700599998
        #   status: OPTIMAL_SOLUTION
        #   sys: 11.412413
        #   user: 68.355111
        # schema: astropy-2.0
        time exptime location.x location.y location.z center.ra center.dec roll
        2012-05-02T18:58:32.699 10.0 -1751.6172079535218 -4316.900611397083 5223.012730741018 115.31249999999999 18.20995686428301 20.0
        2012-05-02T19:08:32.699 10.0 -1056.8666747393759 -215.39372825801107 6912.053809231001 133.59375 12.024699180565822 80.0
        2012-05-02T19:18:32.699 10.0 420.69277073338765 3877.5088670176733 5810.065750855414 136.40625 9.594068226860461 80.0
        2012-05-02T19:29:32.699 10.0 2146.2957309422827 6380.153680388233 1934.1511498095422 146.25 -10.806922874860343 20.0
        2012-05-02T19:39:32.699 10.0 2913.9219997475725 5876.6317556649665 -2468.1843646368943 106.07142857142857 -69.42254649458224 70.0
        2012-05-02T19:49:32.699 10.0 2364.8352599804857 3005.6360601625474 -5875.114354958457 56.25 -60.434438844952275 60.0
        2012-05-02T19:59:32.699 10.0 571.0966946043521 -964.5042611033908 -6920.301799376411 88.04347826086958 -55.87335043525199 0.0
        2012-05-02T20:09:32.699 10.0 -1773.1726479029899 -4368.6218509397195 -5187.571325540394 104.46428571428571 -48.14120779436026 80.0
        2012-05-02T20:19:32.699 10.0 -3621.2261835503236 -5841.333071740781 -1367.7426015403416 69.75 -60.434438844952275 0.0        

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
    <QTable length=9>
              time          exptime ...   roll  found
                              min   ...   deg        
             object         float64 ... float64  bool
    ----------------------- ------- ... ------- -----
    2012-05-02T18:58:32.699    10.0 ...    20.0 False
    2012-05-02T19:08:32.699    10.0 ...    80.0 False
    2012-05-02T19:18:32.699    10.0 ...    80.0 False
    2012-05-02T19:29:32.699    10.0 ...    20.0 False
    2012-05-02T19:39:32.699    10.0 ...    70.0 False
    2012-05-02T19:49:32.699    10.0 ...    60.0 False
    2012-05-02T19:59:32.699    10.0 ...     0.0 False
    2012-05-02T20:09:32.699    10.0 ...    80.0 False
    2012-05-02T20:19:32.699    10.0 ...     0.0  True

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
