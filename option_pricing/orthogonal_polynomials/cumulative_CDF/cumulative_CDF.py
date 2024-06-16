
__name__ = 'option_pricing.orthogonal_polynomials.cumulative_CDF.cumulative_CDF'

from ..CDF.CDF import (
    laguerre as laguerre_CDF,
    legendre as legendre_CDF,
    chebyshev as chebyshev_CDF,
    hermite as hermite_CDF,
    jacobi as jacobi_CDF,
)

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
        ) -> np.ndarray:
    
    power_CDF = lambda x, n: x ** n

    return np.concatenate([power_CDF(n=N, x=x) for N in range(2, n+2)], axis=1)

def laguerre(
        x: np.ndarray,
        n: int, 
        ) -> np.ndarray:
    
    ## Despite traditional polynomials occurring from 0, The Laguerre CDF at 0 returns the constant 1.
    return np.concatenate([laguerre_CDF(n=N, x=x) for N in range(1, n+1)], axis=1)

    ## Reshape and output:
    # return CDFs.T

# Legendre:

def legendre(
        x: np.ndarray,
        n: int, 
        ) -> np.ndarray:

    ## Despite traditional polynomials occurring from 0, The Legendre CDF at 0 returns the constant 1, and at 1 returns itself.
    return np.concatenate(np.c_[[legendre_CDF(n=N, x=x) for N in range(2, n+2)]], axis=1)


# Chebyshev:

def chebyshev(
        x: np.ndarray,
        n: int, 
        ) -> np.ndarray:

    return np.concatenate(np.c_[[chebyshev_CDF(n=N, x=x) for N in range(2, n+2)]], axis=1)

# Hermite:

def hermite(
        x: np.ndarray,
        n: int, 
        ) -> np.ndarray:

    return np.concatenate(np.c_[[hermite_CDF(n=N, x=x) for N in range(2, n+2)]], axis=1)

# Jacobi:

def jacobi(
        x: np.ndarray,
        n: int, 
        alpha: _Union[float, int] = 0, 
        beta: _Union[float, int] = 0, 
        ) -> np.ndarray:

    return np.concatenate(np.c_[[jacobi_CDF(n=N, x=x, alpha=alpha, beta=beta) for N in range(2, n+2)]], axis=1)