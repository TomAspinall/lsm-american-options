import numpy as np

__name__ = 'option_pricing._utils.continuation_value'

from .least_squares_linear_regression import least_squares_linear_regression_fitted_values
from ..orthogonal_polynomials.cumulative_CDF import (
        power,
        laguerre,
        legendre,
        chebyshev,
        hermite,
        jacobi,
)

orthogonal_functions = {"POWER": power,
        "LAGUERRE": laguerre,
        "LEGENDRE": legendre,
        "CHEBYSHEV": chebyshev,
        "HERMITE": hermite,
        "JACOBI": jacobi}

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
    
    orthogonal_function=orthogonal_functions[orthogonal.upper()]

    ## Only in-the-money paths are considered in the LSM regression:
    # Value of waiting:
    continuation_value_in_the_money = continuation_value[in_the_money_paths].copy()
    # Underlying state variables that drive the asset:
    state_variables_t_in_the_money = state_variables_t[in_the_money_paths]

    ## Must be ndarray:
    if state_variables_t_in_the_money.ndim == 1:
        state_variables_t_in_the_money = state_variables_t_in_the_money[:, np.newaxis]

    ## Obtain Orthogonal polynomials:
    regression_matrix = state_variables_t_in_the_money.copy()
    regression_matrix = np.c_[ regression_matrix, orthogonal_function(regression_matrix, degree)]

    ## Cross products:
    ## TODO - Increase efficiency:
    if cross_product and state_variables_t_in_the_money.shape[1] >= 1:
        for i in range(state_variables_t_in_the_money.shape[1]):
            for j in range(i+1, state_variables_t_in_the_money.shape[1]):
                regression_matrix = np.c_[ state_variables_t_in_the_money[:, i] * state_variables_t_in_the_money[:, j], regression_matrix]

    ## The Independent Variables within the least-squares includes the actual values, as well as their orthogonal weights:
    ## Dependent variable is continuation_value_in_the_money:

    ## Perform Least-Squares regression and obtain fitted values:
    A = np.c_[ np.ones(len(regression_matrix)), regression_matrix.copy()]
    continuation_value[in_the_money_paths] = A @ np.linalg.lstsq(A, continuation_value_in_the_money, rcond=None)[0]

    ## Option 2:
    # continuation_value[in_the_money_paths] = least_squares_linear_regression_fitted_values(regression_matrix.copy(), continuation_value_in_the_money)

    return continuation_value