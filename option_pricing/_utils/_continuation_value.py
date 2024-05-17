from numbers import Number
import numpy as np
from math import sqrt, log, ceil
import numpy as np
from math import factorial

# regression_matrix = np.ndarray((4, 19))
# state_variables_ncol = 30


##############################################################################
####################### ESTIMATED CONTINUATION VALUE: ########################
##############################################################################

# in_the_money=in_the_money
# t=t
# continuation_value=american_option_value * discount_rate
# state_variables_t=state_variables[t]
# orthogonal=orthogonal
# degree=degree
# cross_product=cross_product


def estimate_continuation_value(
        in_the_money: np.ndarray,
        t: Number,
        continuation_value,
        state_variables_t,
        orthogonal,
        degree,
        cross_product,
):

    if len(state_variables_t.shape) == 1:
        state_variables_ncol = 1
    else:
        state_variables_ncol = state_variables_t.shape[1]

    ## Only in-the-money paths are considered in the LSM regression:
    state_variables_t_in_the_money = state_variables_t[~np.isnan(state_variables_t)]

    fact_dim = factorial(state_variables_ncol - 1)

    if cross_product and fact_dim > 0:
        number_columns_output = fact_dim
    else:
        number_columns_output = 0

    # index = state_variables_ncol + 1

    # regression_matrix = np.ndarray(shape = (len(continuation_value[in_the_money], len(state_variables_t_in_the_money))))
    # regression.matrix[, range(index)] < - c(continuation_value[in_the_money], state_variables_t_in_the_money)
    # index < - index + 1

    # # Only regress paths In the money (in_the_money):
    # if cross_product:
    #     number_columns = 1 + (state_variables_ncol * degree) + \
    #         factorial(state_variables.ncol - 1)
    #     regression_matrix = np.ndarray(number_columns)
    # else:
    #     number_columns = 1 + (state_variables_ncol * degree)

    # index < - state_variables.ncol+1
    # regression.matrix[, 1:index] < - c(continuation_value[in_the_money], state_variables_t_in_the_money)
    # index < - index + 1
