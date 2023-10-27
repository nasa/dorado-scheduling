Quick Start
===========

To install
----------

To install with `Pip`_:

1.  Run the following command::

        $ pip install git+https://github.com/nasa/dorado-scheduling

To set up the CPLEX optimization engine
---------------------------------------

2.  Set up the CPLEX optimization engine.

    a)  If you are a student, researcher, or faculty member at an academic
        institution, sign up for the `IBM Academic Initiative`_. Download
        and install the free academic version of ILOG CPLEX Optimization
        Studio.

        If running on an HPC like SDSC Expanse, copy the `linux_x86_64`_ installer
        file (e.g. `cplex_studio2211.linux_x86_64.bin`_) to the cluster. Then,
        update the permissions of the file to make it executable and run it 
        with `./cplex_studio2211.linux_x86_64.bin`_. The installation will then
        run in the command line.

        After running the installer, you may need to set the
        `CPLEX_STUDIO_DIR201`_ environment variable so that the CPLEX Python
        interface can locate your licensed copy of CPLEX.

    b)  For all other users, including US Government (e.g. NASA) employees, we
        recommend the `Developer Subscription`_.

        Developer Subscription users do *not* need to run the IBM ILOG CPLEX
        Optimization Studio installer because the necessary components are
        built into the CPLEX Python interface. Simply set the
        `CPLEX_STUDIO_KEY`_ environment variable to the API key that you
        received from IBM.

    For other CPLEX installation scenarios, see the `docplex instructions`_.

To generate an observing plan
-----------------------------

1.  Generate an observing plan for the included example sky map::

        $ dorado-scheduling examples/6.fits -o examples/6.ecsv

    This will take 3 minutes and will use about 10 GB of memory at peak.

2.  Print out the observing plan::

        >>> from astropy.table import QTable
        >>> QTable.read('examples/6.ecsv').pprint_all()
                  time          exptime                location [x, y, z]                                center                roll
                                  min                          km                                       deg,deg                deg 
        ----------------------- ------- ------------------------------------------------ ------------------------------------- ----
        2012-05-02T18:58:32.699    10.0   (-1751.61720795, -4316.9006114, 5223.01273074)  115.31249999999999,18.20995686428301 20.0
        2012-05-02T19:08:32.699    10.0   (-1056.86667474, -215.39372826, 6912.05380923)          133.59375,12.024699180565822 80.0
        2012-05-02T19:18:32.699    10.0     (420.69277073, 3877.50886702, 5810.06575086)           136.40625,9.594068226860461 80.0
        2012-05-02T19:29:32.699    10.0    (2146.29573094, 6380.15368039, 1934.15114981)            146.25,-10.806922874860343 20.0
        2012-05-02T19:39:32.699    10.0   (2913.92199975, 5876.63175566, -2468.18436464) 106.07142857142857,-69.42254649458224 70.0
        2012-05-02T19:49:32.699    10.0   (2364.83525998, 3005.63606016, -5875.11435496)             56.25,-60.434438844952275 60.0
        2012-05-02T19:59:32.699    10.0      (571.0966946, -964.5042611, -6920.30179938)  88.04347826086958,-55.87335043525199  0.0
        2012-05-02T20:09:32.699    10.0  (-1773.1726479, -4368.62185094, -5187.57132554) 104.46428571428571,-48.14120779436026 80.0
        2012-05-02T20:19:32.699    10.0 (-3621.22618355, -5841.33307174, -1367.74260154)             69.75,-60.434438844952275  0.0

3.  To generate an animated visualization for this observing plan, run the
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
    >>> from dorado.scheduling import mission
    >>> healpix = HEALPix(nside=32, frame=ICRS())
    >>> target = SkyCoord(66.91436579*u.deg, -61.98378895*u.deg)
    >>> target_pixel = healpix.skycoord_to_healpix(target)
    >>> schedule = QTable.read('examples/6.ecsv')
    >>> fov = mission.dorado.fov
    >>> footprints = [fov.footprint_healpix(healpix, row['center'], row['roll'])
    ...               for row in schedule]
    >>> schedule['found'] = [target_pixel in footprint for footprint in footprints]
    >>> schedule
    <QTable length=9>
              time          exptime ...   roll  found
                              min   ...   deg        
              Time          float64 ... float64  bool
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
.. _`docplex instructions`: https://ibmdecisionoptimization.github.io/docplex-doc/mp/getting_started.html
.. _`IBM Academic Initiative`: https://www.ibm.com/academic/technology/data-science
.. _`Developer Subscription`: https://www.ibm.com/products/ilog-cplex-optimization-studio/pricing
.. _`CPLEX_STUDIO_DIR201`: https://www.ibm.com/support/pages/entering-your-api-key-and-setting-cplexstudiokey-environment-variable
.. _`CPLEX_STUDIO_KEY`: https://www.ibm.com/support/pages/entering-your-api-key-and-setting-cplexstudiokey-environment-variable
