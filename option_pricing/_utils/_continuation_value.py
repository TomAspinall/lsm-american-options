import numpy as np

from .._utils._least_squares_linear_regression import least_squares_linear_regression_fitted_values
from ..orthogonal_polynomials.cumulative_CDF import (
        power,
        laguerre,
        legendre,
        chebyshev,
        hermite,
        jacobi,
)

from option_pricing._utils._least_squares_linear_regression import least_squares_linear_regression_fitted_values
from option_pricing.orthogonal_polynomials.cumulative_CDF import (
        power,
        laguerre,
        legendre,
        chebyshev,
        hermite,
        jacobi,
)

##############################################################################
####################### ESTIMATED CONTINUATION VALUE: ########################
##############################################################################

def estimate_continuation_value(
        in_the_money_paths: np.ndarray,
        continuation_value: np.array,
        state_variables_t: np.ndarray,
        orthogonal: str,
        degree: int,
        cross_product: bool,
):
    
    orthogonal_function = {"POWER": power,
        "LAGUERRE": laguerre,
        "LEGENDRE": legendre,
        "CHEBYSHEV": chebyshev,
        "HERMITE": hermite,
        "JACOBI": jacobi}[orthogonal.upper()]

    ## Only in-the-money paths are considered in the LSM regression:
    # Value of waiting:
    continuation_value_in_the_money = continuation_value[in_the_money_paths]
    # Underlying state variables that drive the asset:
    state_variables_t_in_the_money = state_variables_t[in_the_money_paths]

    ## Must be ndarray:
    if state_variables_t_in_the_money.ndim == 1:
        state_variables_t_in_the_money = state_variables_t_in_the_money[:, np.newaxis]

    ## Cross products:
    regression_matrix = state_variables_t_in_the_money.copy()
    ## TODO - Increase efficiency:
    if cross_product and state_variables_t_in_the_money.shape[1] > 1:
        for i in range(state_variables_t_in_the_money.shape[1]):
            for j in range(i+1, state_variables_t_in_the_money.shape[1]):
                regression_matrix = np.c_[ state_variables_t_in_the_money[:, i] * state_variables_t_in_the_money[:, j], regression_matrix]

    ## Obtain Orthogonal polynomials:
    regression_matrix = np.c_[ regression_matrix, orthogonal_function(regression_matrix, degree)]

    ## The Independent Variables within the least-squares includes the actual values, as well as their orthogonal weights:
    ## Dependent variable is continuation_value_in_the_money:
    continuation_value[in_the_money_paths] = least_squares_linear_regression_fitted_values(regression_matrix, continuation_value_in_the_money)

    return continuation_value