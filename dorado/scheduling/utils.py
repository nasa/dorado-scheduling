#
# Copyright © 2021 United States Government as represented by the Administrator
# of the National Aeronautics and Space Administration. No copyright is claimed
# in the United States under Title 17, U.S. Code. All Other Rights Reserved.
#
# SPDX-License-Identifier: NASA-1.3
#
"""Miscellaneous utilities."""
import numpy as np


def nonzero_intervals(a):
    """Find the intervals over which an array is nonzero.

    Parameters
    ----------
    a : numpy.ndarray
       A 1D array of length `N`.

    Returns
    -------
    indices : numpy.ndarray
       An array of size (`N`, 2) denoting the start and end indices of the
       intervals with contiguous nonzero values in `a`.

    Examples
    --------
    >>> nonzero_intervals([])
    array([], shape=(0, 2), dtype=int64)
    >>> nonzero_intervals([0, 0, 0, 0])
    array([], shape=(0, 2), dtype=int64)
    >>> nonzero_intervals([1, 1, 1, 1])
    array([[0, 3]])
    >>> nonzero_intervals([0, 1, 1, 1])
    array([[1, 3]])
    >>> nonzero_intervals([1, 1, 1, 0])
    array([[0, 2]])
    >>> nonzero_intervals([1, 1, 0, 1, 0, 1, 1, 1])
    array([[0, 1],
           [3, 3],
           [5, 7]])

    """
    a = np.pad(np.asarray(a, dtype=bool), 1)
    return np.column_stack((np.flatnonzero(a[1:-1] & ~a[:-2]),
                            np.flatnonzero(a[1:-1] & ~a[2:])))
