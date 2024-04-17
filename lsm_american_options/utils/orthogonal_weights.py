from orthogonal_polynomials import (
    orthogonal_weight_chebyshev,
    orthogonal_weight_hermite,
    orthogonal_weight_jacobi,
    orthogonal_weight_laguerre,
    orthogonal_weight_legendre,
)
from math import floor, factorial

##############################################################################
########################## Orthogonal Algorithms: ############################
##############################################################################

orthogonal_options = ["POWER", "LAGUERRE",
                      "LEGENDRE", "CHEBYSHEV", "HERMITE", "JACOBI"]


def orthogonal_weight(
        n: int,
        x,
        orthogonal: str = "Laguerre",
        alpha: float = 0.0,
        beta: float = 0.0) -> list:

    assert orthogonal.upper() in orthogonal_options, f"arg 'orthogonal' expected one of: {
        orthogonal_options}"

    # Case insensitive:
    orthogonal = orthogonal.upper()

    # n or n / 2 ?
    if orthogonal in ["LEGENDRE", "CHEBYSHEV", "HERMITE"]:
        N = n / 2
    else:
        N = n

    if N % 1 != 0:
        Warning(f"Orthogonal weighting (n) rounded down from {n} to {n-1}")
    N = floor(N)

    # Early return - Power function:
    if orthogonal == "POWER":
        return x ** N

    # Iterative Orthogonal Weights:

    if orthogonal == "LAGUERRE":
        return sum((orthogonal_weight_laguerre(N, m, x) for m in range(N+1)))

    elif orthogonal == "LEGENDRE":
        return (1 / 2 ** n) * sum((orthogonal_weight_legendre(N, m, x) for m in range(N+1)))

    elif orthogonal == "CHEBYSHEV":
        return (N / 2) * sum((orthogonal_weight_chebyshev(N, m, x) for m in range(N+1)))

    elif orthogonal == "HERMITE":
        return factorial(N) * sum((orthogonal_weight_hermite(N, m, x) for m in range(N+1)))

    elif orthogonal == "JACOBI":
        return (1 / 2 ** n) * sum((orthogonal_weight_jacobi(N, m, alpha, beta, x) for m in range(N+1)))
