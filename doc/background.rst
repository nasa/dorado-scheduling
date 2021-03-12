Background
==========

Given a gravitational-wave HEALPix probability sky map, this Python package
finds an optimal sequence of Dorado observations to maximize the probability of
observing the (unknown) location of the gravitational-wave event, within one
orbit.

The problem is formulated as a mixed integer programming problem with the
following arrays of binary decision variables:

* ``schedule`` (``npix`` × ``nrolls`` × ``ntimes - ntimes_per_exposure + 1``):
  1 if an observation of the field that is centered on the given HEALPix pixel,
  at a given roll angle, is begun on a given time step; or 0 otherwise
* ``pix`` (``npix``): 1 if the given HEALPix pixel is observed, or 0
  otherwise

The problem has the following parameters:

* ``nexp`` (scalar, integer): the maximum number of exposures
* ``prob`` (``npix``, float): the probability sky map
* ``regard`` (``npix`` × ``ntimes``, binary): 1 if the field centered on
  the given HEALPix pixel is within the field of regard at the given time, 0
  otherwise

The objective function is the sum over all of the pixels in the probability sky
map for which the corresponding entry in ``pix`` is 1.

The constraints are:

* At most one observation is allowed at a time.
* At most ``nexp`` observations are allowed in total.
* A given pixel is observed if any field that contains it within its
  footprint is observed.
* A field may be observed only if it is within the field of regard.
