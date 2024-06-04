from math import factorial as _factorial
from math import comb as  _comb

##############################################################################
################# Orthogonal Probability Density Functions: ##################
##############################################################################

"""
probability density functions of available orthogonal polynomials.

Possible orthogonal PDF's:
    POWER
    LAGUERRE
    LEGENDRE
    CHEBYSHEV
    HERMITE
    JACOBI
"""


def laguerre(n, m, x):
    c_x = ((-1) ** m) * _comb(n, n-m) * (1/_factorial(m))
    g_m_x = x ** m
    return c_x * g_m_x

# Legendre:

def legendre(n, m, x):

    c_x = (-1 ** m) * _comb(n, m) * _comb(2*n - 2*m, n)
    g_m_x = x ** (n - 2 * m)
    return c_x * g_m_x

# Chebyshev:

def chebyshev(n, m, x):
    c_x = (-1 ** m) * _factorial(n - m - 1) / \
        (_factorial(m) * _factorial(n - 2 * m))
    g_m_x = (2 * x) ** (n - (2 * m))
    return c_x * g_m_x

# Hermite:

def hermite(n, m, x):
    c_x = (1 ** m) / (_factorial(m) * (2 ** m) * (_factorial(n - 2 * m)))
    g_m_x = x ** (n - 2 * m)
    return c_x * g_m_x

# Jacobi:

def jacobi(n, m, x, alpha, beta):
    c_x = _comb(n + alpha, m) * _comb(n + beta, n - m)
    g_m_x = ((x - 1) ** (n - m)) * ((x + 1) ** m)
    return c_x * g_m_x

