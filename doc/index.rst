.. dorado-scheduling documentation master file, created by
   sphinx-quickstart on Fri Mar 12 09:25:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Dorado Scheduling
=================

Dorado is a proposed space mission for ultraviolet follow-up of gravitational
wave events. This project is a simple target of opportunity observation planner
for Dorado.


Features
--------

* **Global**: jointly and globally solves the problems of tiling (the set of
  telescope boresight orientations and roll angles) and the scheduling (which
  tile is observed at what time), rather than solving each sub-problem one at
  a time
* **Optimal**: generally solves all the way to optimality, rather than
  finding merely a "good enough" solution
* **Fast**: solve an entire orbit in about 5 minutes
* **General**: does not depend on heuristics of any kind
* **Flexible**: problem is formulated in the versatile framework of
  `mixed integer programming`_


Contents
--------

.. toctree::
   :maxdepth: 1

   background
   quickstart
   reference/index

.. image:: _static/6.gif
   :alt: Example Dorado observing plan

.. _`mixed integer programming`: https://en.wikipedia.org/wiki/Integer_programming
