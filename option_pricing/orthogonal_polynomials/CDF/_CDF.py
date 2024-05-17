from ..PDF._PDF import (
    laguerre as laguerre_PDF,
    legendre as legendre_PDF,
    chebyshev as chebyshev_PDF,
    hermite as hermite_PDF,
    jacobi as jacobi_PDF,
)
from math import floor as _floor
from math import factorial as _factorial
from numpy import array as _np_array
from numpy import ndarray as _np_ndarray
from typing import Union as _Union

##############################################################################
########################## Orthogonal Algorithms: ############################
##############################################################################

"""
..PDF.orthogonals displays the probability density functions of available orthogonal polynomials.
Outputs used in least-squares Monte Carlo simulation, however, are typically cumulative density funcions with a given level of weighting.
Note that the weighting of the orthogonal CDF utilised will impact processing time, but asymptotically decrease bias / increase accuracy to the true option value.
For more detail, see: Longstaff, Francis A., and Eduardo S. Schwartz. "Valuing American options by simulation: a simple least-squares approach." The review of financial studies 14.1 (2001): 113-147.
"""

"""
Possible orthogonal CDF's:
    POWER
    LAGUERRE
    LEGENDRE
    CHEBYSHEV
    HERMITE
    JACOBI
"""

# Power (early return, no cumulation required):

def power(
        n: int,
        x: _Union[list, _np_array, _np_ndarray], 
        ) -> list:
    
    ## N = n:
    N = n

    ## d_n multiplier:
    d_n = 1

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## Unique (no cumulation / early return:)
    return d_n * (x ** N)


# Laguerre - CDF:

def laguerre(
        n: int, 
        x: _Union[list, _np_array, _np_ndarray],
        ) -> list:

    ## Iteration for polynomial:
    N = n

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## d_n multiplier:
    d_n = 1

    ## Perform cumulation:
    return d_n * sum((laguerre_PDF(n, m, x) for m in range(N+1)))


# Legendre:

def legendre(
        n: int, 
        x: _Union[list, _np_array, _np_ndarray],
        ) -> list:

    ## Iteration for polynomial:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## d_n multiplier:
    d_n = 1 / 2 ** n

    ## Perform cumulation:
    return d_n * sum((legendre_PDF(n, m, x) for m in range(N+1)))


# Chebyshev:

def chebyshev(
        n: int, 
        x: _Union[list, _np_array, _np_ndarray],
        ) -> list:

    ## Iteration for polynomial:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## d_n multiplier:
    d_n = n / 2

    ## Perform cumulation:
    return d_n * sum((chebyshev_PDF(n, m, x) for m in range(N+1)))


# Hermite:

def hermite(
        n: int, 
        x: _Union[list, _np_array, _np_ndarray],
        ) -> list:

    ## Utilises N/2:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## d_n multiplier:
    d_n = _factorial(N)

    ## Perform cumulation:
    return d_n * sum((hermite_PDF(N, m, x) for m in range(N+1)))


# Jacobi:

def jacobi(
        n: int, 
        x: _Union[list, _np_array, _np_ndarray],
        alpha: float = 0.0,
        beta: float = 0.0,
        ) -> list:

    ## Iteration for polynomial:
    N = n

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {n} to {N}")

    ## d_n multiplier:
    d_n = 1 / 2 ** n

    ## Perform cumulation:
    return d_n * sum((jacobi_PDF(n, m, x) for m in range(N+1)))