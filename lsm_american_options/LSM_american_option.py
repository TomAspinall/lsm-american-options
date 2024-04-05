from math import exp, sqrt
from statistics import mean, variance
import pandas as pd
import numpy as np
number_simulations = 100
number_periods = 10

discount = lambda r, t=1: exp(-r*t)

def LSM_american_option(state_variables, 
                        payoff, 
                        K, 
                        dt, 
                        rf, 
                        call = False, 
                        orthogonal = "Power", 
                        degree = 2, 
                        cross_product = True, 
                        verbose = False,
                        **kwargs,
                        ):

    ## Helper:
    discount = lambda r, t=1: exp(-r*t)

    ## Number simulations = Number columns:
    number_simulations = len(state_variables.columns)

    ### Assertions:
    assert len(K) > 1 and len(K) != number_simulations, "length of object 'K' does not equal 1 or number of columns of 'state_variables'!"
    assert state_variables.notna().all(), "NA's have been specified within 'state_variables'!"
    assert (len(state_variables) != len(payoff)), "Dimensions of object 'state_variables' does not match 'payoff'!"

    # Nominal interest rate:
    r = rf * dt

    # Number of discrete time periods:
    number_periods = len(state_variables)

    ##############################################################################
    ######################## Initialise Memory Objects: ##########################
    ##############################################################################

    ## Profit - Value of immediate exercise:
    profit = np.ndarray(shape = (number_simulations, number_periods))

    ## Continuation value - expected value of waiting to exercise. Compared with immediate profit to make exercise decisions through dynamic programming:
    continuation_value = [0 for x in range(number_simulations)]

    ## American option value of project, given that you can either delay or exercise:
    american_option_value = [0 for x in range(number_simulations)]

    ## Optimal period of exercise is the earliest time that exercise is triggered. If no exercise, an NA is returned:
    exercise_timing = [None for x in range(number_simulations)]

    ##############################################################################
    ###################### Begin LSM Simulation Algorithm: #######################
    ##############################################################################


    ## Backwards induction begin:
    for t in range(number_periods, 0, -1):
        # t is representative of the index for time, but in reality the actual time period you're in is (t-1).
        
        ## STEP ONE:
        ### Forward foresight (high bias) - the Immediate payoff of exercising:
        if(call):
            profit[t,] = pmax(payoff[t,] - K, 0)
        else:
            profit[t,] = pmax(K - payoff[t,], 0)
        
        ## STEP TWO:
        ### Estimated continuation values:
        if t < number_periods:

            #We only consider the invest / delay investment decision for price paths that are in the money (ie. positive NPV):
            in_the_money = which(profit[t,] > 0)

            #Only regress paths In the money (in_the_money):
            if len(in_the_money)>0:
                continuation_value = continuation_value_calc(
                                                            in_the_money = in_the_money,
                                                            t = t,
                                                            continuation_value = american_option_value  * discount(r),
                                                            state_variables_t = state_variables[t,,],
                                                            orthogonal = orthogonal,
                                                            degree = degree,
                                                            cross_product = cross_product)
            else:
                continuation_value = american_option_value  * discount(r)


        ## STEP THREE:
        ### Dynamic programming:
        exercise = profit[t,] > continuation_value

        ## Discount existing values if not exercising
        american_option_value[~exercise] <- american_option_value[~exercise] * discount(r)
        ## Receive immediate profit if exercising
        american_option_value[exercise] <- profit[t,exercise]

        ## Was the option exercised?
        exercise_timing[exercise] <- t

        ### Re-iterate.
    ### End backwards induction.

    ## Calculate project value -  discount payoffs
    option_values = [0 for x in number_simulations]


    ## American option value - discounting payoffs back to time zero, averaging over all paths.

    exercised_paths = exercise_timing.notna()
    exercise_period = exercise_timing[exercised_paths]
    in_the_money_paths = [x for x in exercised_paths if x]
    option_values[exercised_paths] <- profit[number_periods * (in_the_money_paths-1) + exercise_period] * discount(r, exercise_period)

    ## Calculate option value:
    option_price = mean(option_values)

    ## Typical return:
    verbose = kwargs.get('verbose', None)
    if verbose:
        return option_price
    ## Verbose / debug - log it all:
    elif verbose is True:
        ## Verbose Outputs:
        exercise_time = (exercise_period - 1) * dt

        return {
            ## Option value
            "Value" : option_price,
            ## Option value standard error
            "Standard Error" : sqrt(variance(option_values) / number_simulations),
            ## expected exercise time
            "Expected Timing" : mean(exercise_time),
            ## exercise time standard error
            "Expected Timing SE" : sqrt(variance(exercise_time) / number_simulations),
            ## exercise prob.
            "Exercise Probability" : len(in_the_money_paths) / number_simulations,
            ## cumulative exercise prob.
            "Cumulative Exercise Probability" : cumsum(table(c(exercise_time, (0:(number_periods - 1)) * dt)) - 1) / number_simulations
        }
