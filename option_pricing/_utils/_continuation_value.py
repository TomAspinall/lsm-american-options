from numbers import Number
import numpy as np
from math import sqrt, log, ceil
import numpy as np
from math import factorial

##############################################################################
####################### ESTIMATED CONTINUATION VALUE: ########################
##############################################################################

def estimate_continuation_value(
        profit_t: np.ndarray,
        in_the_money_paths: np.ndarray,
        continuation_value: np.array,
        state_variables_t: np.ndarray,
        t: Number,
        orthogonal: str,
        degree: int,
        cross_product: bool,
):

    ## Only in-the-money paths are considered in the LSM regression:
    # Profit:
    profit_in_the_money = profit_t[in_the_money_paths]
    # Value of waiting:
    continuation_value_in_the_money = continuation_value[in_the_money_paths]
    # Underlying state variables that drive the asset:
    state_variables_t_in_the_money = state_variables_t[in_the_money_paths]

    # state_variables_ncol = state_variables_t.shape[1]
    # state_variables_ncol = 1
    # fact_dim = factorial(state_variables_ncol - 1)

    # if cross_product and fact_dim > 0:
    #     number_columns_output = fact_dim
    # else:
    #     number_columns_output = 0
    
    ## The Independent Variables within the least-squares includes the actual values, as well as their orthogonal weights:
    # regression_matrix = np.zeros()
    ## Dependent variable is continuation_value_in_the_money:
    y = continuation_value_in_the_money
    x = state_variables_t_in_the_money

    ## Assemble matrix A:
    independent_variables = [x, ]
    A = np.vstack(independent_variables).T
    # turn y into a column vector:
    y = y[:, np.newaxis]

    ## Direct least-square regression:
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y)

    ## Attain fitted values:
    number_independent_variables = A.shape[1]
    fitted_values = np.zeros(A.shape[0])
    for i in range(number_independent_variables):
        fitted_values += x[i] * alpha[i]
    
    continuation_value[in_the_money_paths] = fitted_values
    return continuation_value