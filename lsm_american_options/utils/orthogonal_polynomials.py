from math import factorial, comb

##############################################################################
########################## Orthogonal Algorithms: ############################
##############################################################################

"""
Possible Orthogonal Polynomials:
Power
Laguerre
Legendre
Chebyshev
Hermite
Jacobi
"""

# Laguerre:


def orthogonal_weight_laguerre(n, m, x):
    c_x = (-1 ** m) * comb(n, n-m) * (1/factorial(m))
    g_m_x = x ** m
    return c_x * g_m_x

# Legendre:


def orthogonal_weight_legendre(n, m, x):
    c_x = (-1 ** m) * comb(n, m) * comb(2*n - 2*m, n)
    g_m_x = x ** (n - 2 * m)
    return c_x * g_m_x

# Chebyshev:


def orthogonal_weight_chebyshev(n, m, x):
    c_x = (-1 ** m) * factorial(n - m - 1) / \
        (factorial(m) * factorial(n - 2 * m))
    g_m_x = (2 * x) ** (n - (2 * m))
    return c_x * g_m_x

# Hermite:


def orthogonal_weight_hermite(n, m, x):
    c_x = (1 ** m) / (factorial(m) * (2 ** m) * (factorial(n - 2 * m)))
    g_m_x = x ** (n - 2 * m)
    return c_x * g_m_x

# Jacobi:


def orthogonal_weight_jacobi(n, m, alpha, beta, x):
    c_x = comb(n + alpha, m) * comb(n + beta, n - m)
    g_m_x = ((x - 1) ** (n - m)) * ((x + 1) ** m)
    return c_x * g_m_x
