import numpy as np
from numbers import Number

# Requirements:
from .._utils._continuation_value import estimate_continuation_value
from .._utils._discount import discount


def least_squares_monte_carlo(state_variables: np.ndarray,
                        payoff: np.ndarray,
                        K: Number | np.ndarray,
                        delta_time: Number,
                        risk_free_rate: Number,
                        call: bool = False,
                        orthogonal: str = "Power",
                        degree: int = 2,
                        cross_product: bool = True,
                        ):

    ## Const:

    # length rolumns: # simulations
    # length rows:    # discrete time periods
    number_periods, number_simulations = state_variables.shape

    # Nominal interest rate:
    nominal_interest_rate = risk_free_rate * delta_time
    # Corresponding discount:
    discount_rate = discount(nominal_interest_rate)

    # Time period of backwards induction:
    termination_period = number_periods - 1

    # Assertions:
    # assert type(K) is Number and not len(K) == number_simulations, "length of object 'K' does not equal 1 or number of columns of 'state_variables'!"
    assert not np.isnan(state_variables).any(), "NA's have been specified within 'state_variables'!"
    assert state_variables.shape == payoff.shape, "Dimensions of object 'state_variables' does not match 'payoff'!"

    ## Safety:
    # if len(state_variables.shape) == 2:
    #     the_shape = state_variables.shape
    #     state_variables = np.ndarray((the_shape[0], the_shape[1], 1))

    ##############################################################################
    ######################## Initialise Memory Objects: ##########################
    ##############################################################################

    # Profit - Value of immediate exercise:
    profit = np.zeros(state_variables.shape)

    # Continuation value - expected value of waiting to exercise. Compared with immediate profit to make exercise decisions through dynamic programming:
    continuation_value = np.zeros(number_simulations)

    # American option value of project, given that you can either delay or exercise:
    american_option_value = np.empty(number_simulations)
    american_option_value[:] = np.nan

    # Optimal period of exercise is the earliest time that exercise is triggered. If no exercise, an NA is returned:
    exercise_timing = np.empty(number_simulations)
    exercise_timing[:] = np.nan

    ##############################################################################
    ###################### Begin LSM Simulation Algorithm: #######################
    ##############################################################################

    # STEP ONE:
    # Forward foresight (high bias) - the Immediate payoff of exercising:
    if call:
        profit[termination_period] = np.maximum(payoff[termination_period] - K, 0)
    else:
        profit[termination_period] = np.maximum(K - payoff[termination_period], 0)


    # Backwards induction begin:
    t = termination_period
    for t in range(termination_period, 1, -1):
        # t is representative of the index for time, but in reality the actual time period you're in is (t-1).

      
        # STEP TWO:
        # Estimated continuation values:

        # We only consider the invest / delay investment decision for price paths that are in the money (ie. positive cash flow):
        in_the_money = profit[t, profit[t] > 0]


        # Only regress paths In the money (in_the_money):
        if len(in_the_money) > 0:
            continuation_value = estimate_continuation_value(
                in_the_money=in_the_money,
                t=t,
                continuation_value=american_option_value * discount_rate,
                state_variables_t=state_variables[t],
                orthogonal=orthogonal,
                degree=degree,
                cross_product=cross_product)
        else:
            continuation_value = american_option_value * discount_rate

        # STEP THREE:
        # Dynamic programming:
        exercise = profit[t,] > continuation_value

        # Discount existing values if not exercising
        american_option_value[~exercise] < - \
            american_option_value[~exercise] * discount_rate
        # Receive immediate profit if exercising
        american_option_value[exercise] < - profit[t, exercise]

        # Was the option exercised?
        exercise_timing[exercise] < - t

        # Re-iterate.
    # End backwards induction.

    # Calculate project value -  discount payoffs
    option_values = np.repeat(0, number_simulations)

    # American option value - discounting payoffs back to time zero, averaging over all paths.

    exercised_paths = exercise_timing.notna()
    exercise_period = exercise_timing[exercised_paths]
    in_the_money_paths = [x for x in exercised_paths if x]
    option_values[exercised_paths] < - profit[number_periods *
                                              (in_the_money_paths-1) + exercise_period] * discount(nominal_interest_rate, exercise_period)

    # Calculate option value:
    option_price = mean(option_values)

    # Execise time:
    exercise_time = (exercise_period - 1) * delta_time

    return option_results(option_price, option_values, number_simulations, exercise_time, in_the_money_paths, delta_time)
