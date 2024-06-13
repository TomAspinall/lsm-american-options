
# __name__ = 'option_pricing.orthogonal_polynomials.cumulative_CDF.CDF'

from ..PDF.PDF import (
    laguerre as laguerre_PDF,
    legendre as legendre_PDF,
    chebyshev as chebyshev_PDF,
    hermite as hermite_PDF,
    jacobi as jacobi_PDF,
)

from math import floor as _floor
from math import factorial as _factorial
import numpy as np
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
        x: np.ndarray, 
        n: int,
        ) -> list:

    if n > 1:
        ## Unique (no cumulation / early return:)
        return np.concatenate([x ** m for m in range(2, n + 1)], axis=1)
    else:
        return x

# Laguerre:

def laguerre(
        x: np.ndarray,
        n: int, 
        ) -> list:

    ## Iteration for polynomial:
    N = n

    ## PDF's:
    PDFs = np.c_[[laguerre_PDF(n, m, x) for m in range(1, N+1)]]
    ## CDFs:
    CDFs = np.cumsum(PDFs, axis=0)
    ## Reshape and output:
    return CDFs.reshape(CDFs.shape[0] * CDFs.shape[2], CDFs.shape[1]).T   


# Legendre:

def legendre(
        x: np.ndarray,
        n: int, 
        ) -> list:

    ## Iteration for polynomial:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {N+1} to {N}")
    N = int(N)

    ## PDF's:
    PDFs = np.c_[[legendre_PDF(n, m, x) for m in range(1, N+1)]]
    ## CDFs:
    CDFs = np.cumsum(PDFs, axis=0)
    ## Reshape:
    CDFs = CDFs.reshape(CDFs.shape[0] * CDFs.shape[2], CDFs.shape[1]).T   
    ## d_n multiplier:
    for i in range(CDFs.shape[1]):
        d_n = 1 / 2 ** n
        CDFs[:,i] *= d_n

    ## Perform cumulation:
    return CDFs


# Chebyshev:

def chebyshev(
        x: np.ndarray,
        n: int, 
        ) -> list:

    ## Iteration for polynomial:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {N+1} to {N}")
    N = int(N)

    ## PDF's:
    PDFs = np.c_[[chebyshev_PDF(n, m, x) for m in range(1, N+1)]]
    ## CDFs:
    CDFs = np.cumsum(PDFs, axis=0)
    ## Reshape:
    CDFs = CDFs.reshape(CDFs.shape[0] * CDFs.shape[2], CDFs.shape[1]).T   
    ## d_n multiplier:
    for i in range(CDFs.shape[1]):
        d_n = i / 2
        CDFs[:,i] *= d_n

    ## Perform cumulation:
    return CDFs


# Hermite:

def hermite(
        x: np.ndarray,
        n: int, 
        ) -> list:

    ## Utilises N/2:
    N = n / 2

    ## Catch:
    if N % 1 != 0:
        N = _floor(N)
        Warning(f"Orthogonal weighting parameter (n) rounded down from {N+1} to {N}")
    N = int(N)

    ## PDF's:
    PDFs = np.c_[[hermite_PDF(n, m, x) for m in range(1, N+1)]]
    ## CDFs:
    CDFs = np.cumsum(PDFs, axis=0)
    ## Reshape:
    CDFs = CDFs.reshape(CDFs.shape[0] * CDFs.shape[2], CDFs.shape[1]).T   
    ## d_n multiplier:
    for i in range(CDFs.shape[1]):
        d_n = _factorial(i)
        CDFs[:,i] *= d_n

    return CDFs


# Jacobi:

def jacobi(
        x: np.ndarray,
        n: int, 
        alpha: _Union[float, int] = 0, 
        beta: _Union[float, int] = 0, 
        ) -> list:

    ## Iteration for polynomial:
    N = n

    ## PDF's:
    PDFs = np.c_[[jacobi_PDF(n, m, x, alpha, beta) for m in range(1, N+1)]]
    ## CDFs:
    CDFs = np.cumsum(PDFs, axis=0)
    ## Reshape:
    CDFs = CDFs.reshape(CDFs.shape[0] * CDFs.shape[2], CDFs.shape[1]).T   
    ## d_n multiplier:
    for i in range(CDFs.shape[1]):
        d_n = 1 / 2 ** i
        CDFs[:,i] *= d_n

    ## Perform cumulation:
    return CDFs