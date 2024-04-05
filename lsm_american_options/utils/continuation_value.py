import numpy as np
from math import factorial

# regression_matrix = np.ndarray((4, 19))
# state_variables_ncol = 30



##############################################################################
####################### ESTIMATED CONTINUATION VALUE: ########################
##############################################################################

def estimate_continuation_value(
        in_the_money, 
        t, 
        continuation_value, 
        state_variables_t, 
        orthogonal, 
        degree, 
        cross_product
        ):

    ## We only consider the invest / delay investment decision for price paths that are in the money (ie. positive NPV):
    in_the_money_length = len(in_the_money)

    ## Early return:
    if in_the_money_length == 0:
        return continuation_value
    
    state_variables_ncol = len(state_variables.columns())
    state_variables_t_in_the_money = [x for x in state_variables_t if state_variables_t is not None]

    if cross_product and factorial(state_variables_ncol - 1) > 0:
        number_columns_output = factorial(state_variables_ncol - 1)
    else:
        number_columns_output = 0

    index <- len(state_variables.columns) + 1
    regression.matrix[,1:index] <- c(continuation_value[in_the_money], state_variables_t_in_the_money)
    index <- index + 1

    #Only regress paths In the money (in_the_money):
    if cross_product:
        number_columns = 1 + state_variables_ncol * degree + factorial(state_variables.ncol - 1)
        regression_matrix = np.ndarray(in_the_money_length, 1 + state_variables_ncol * degree + cross_product)
    else:
        regression_matrix = np.ndarray(in_the_money_length, (1 + state_variables_ncol * degree + ifelse(cross_product, factorial(state_variables.ncol - 1),0)))


    

    regression_matrix <- np.ndarray((in_the_money_length, (1 + state_variables_ncol * degree + ifelse(cross_product, factorial(state_variables.ncol - 1),0)))
    index <- state_variables.ncol+1
    regression.matrix[,1:index] <- c(continuation_value[in_the_money], state_variables_t_in_the_money)
    index <- index + 1



    pass