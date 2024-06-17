import numpy as np

# __name__ = 'option_pricing._utils.continuation_value'

from numpy.polynomial import (
    polynomial,
    laguerre,
    chebyshev,
    legendre,
    hermite
)

## Available Numpy implementations of orthogonal polynomials:
orthogonal_functions = {
    "POWER": polynomial.polyval,
    "LAGUERRE": laguerre.lagval,
    "LEGENDRE": legendre.legval,
    "CHEBYSHEV": chebyshev.chebval,
    "HERMITE": hermite.hermval,
    }

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

    ## Independendent variables:
    regression_matrix = np.c_[
        ## Obtain an intercept and the first n orthogonal polynomials:
        np.concatenate(orthogonal_function(state_variables_t_in_the_money, np.eye(degree+1)), axis=1),
        ## The state variables themselves are included within the regression:
        state_variables_t_in_the_money
    ]

    ## Append cross products:
    number_state_variables = state_variables_t_in_the_money.shape[1]
    if cross_product and number_state_variables > 1:
        regression_matrix = np.concatenate([regression_matrix] + \
                                                [ state_variables_t_in_the_money[:, i] * state_variables_t_in_the_money[:, j] \
                                                for i in range(number_state_variables) \
                                                for j in range(i+1, number_state_variables)
                                                ]
                                            ,axis=1
                                            )

    ## Perform Least-Squares regression and obtain fitted values:
    continuation_value[in_the_money_paths] = regression_matrix @ np.linalg.lstsq(regression_matrix, continuation_value_in_the_money, rcond=None)[0]

    return continuation_value