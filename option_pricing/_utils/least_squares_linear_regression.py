import numpy
import scipy
from scipy import linalg

## Perform least squares linear regression and return fitted values:
def least_squares_linear_regression_fitted_values(
        independent_variables: numpy.ndarray, 
        dependent_variable: numpy.array
        ) -> numpy.array:

    ## Assemble matrix A:
    A = numpy.c_[ numpy.ones(len(independent_variables)), independent_variables]
    # turn dependent variable into a column vector:
    dependent_variable = dependent_variable[:, numpy.newaxis]

    ## Direct least-square regression:
    dot_product = A.T @ A
    if numpy.linalg.det(dot_product) > 0:
        # inv_matrix = numpy.linalg.inv(dot_product)
        inv_matrix = scipy.linalg.inv(dot_product)        
    else:
        # inv_matrix = numpy.linalg.pinv(dot_product)
        inv_matrix = scipy.linalg.pinv(dot_product)
    
    ## Estimated regressors:
    alpha = inv_matrix @ A.T @ dependent_variable

    ## Fitted values:
    fitted_values = A @ alpha

    return fitted_values[:,0]

