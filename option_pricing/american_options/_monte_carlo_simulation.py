import numpy
from numbers import Number
from typing import Optional, Union

# Requirements:
from .._utils._continuation_value import estimate_continuation_value
from .._utils._discount import discount

from option_pricing._utils._continuation_value import estimate_continuation_value
from option_pricing._utils._discount import discount

n = int(1e4)
t = 1
S0 = 40
# risk_free_rate = 0.06
risk_free_rate = 0.0
sigma = 0.0001
time_step = 1/50

from option_pricing.stochastic_differential_equations._geometric_brownian_motion import geometric_brownian_motion as GBM
# Step 1 - Simulate stock prices:
stock_prices = GBM(n, t, risk_free_rate, sigma, S0, time_step)

state_variables = stock_prices
payoff = stock_prices
strike_price = S0
call = False
orthogonal = "Power"
degree = 2
cross_product = True


def monte_carlo_simulation(state_variables: numpy.ndarray,
                        payoff: numpy.ndarray,
                        strike_price: Union[Number, numpy.ndarray],
                        time_step: Number,
                        risk_free_rate: Number,
                        call: Optional[bool] = True,
                        orthogonal: Optional[str] = "Power",
                        degree: Optional[int] = 2,
                        cross_product: Optional[bool] = True,
                        ):

    ## Singular simulated state variable:
    # if len(state_variables.shape) == 2:
    #     state_variables = state_variables.reshape(state_variables.shape + (1,))

    ## Const:

    # length rolumns: # simulations
    # length rows:    # discrete time periods
    # length slices:  # underlying state variables
    # number_periods, number_simulations, number_state_variables = state_variables.shape
    number_periods, number_simulations = state_variables.shape

    # Nominal interest rate:
    nominal_interest_rate = risk_free_rate * time_step
    # Corresponding discount:
    discount_rate = discount(nominal_interest_rate)

    # Time period of backwards induction:
    termination_period = number_periods - 1

    # Assertions:
    # assert type(K) is Number and not len(K) == number_simulations, "length of object 'K' does not equal 1 or number of columns of 'state_variables'!"
    assert not numpy.isnan(state_variables).any(), "NA's have been specified within 'state_variables'!"
    # assert state_variables.shape == payoff.shape, "Dimensions of object 'state_variables' does not match 'payoff'!"

    ## Safety:
    # if len(state_variables.shape) == 2:
    #     the_shape = state_variables.shape
    #     state_variables = numpy.ndarray((the_shape[0], the_shape[1], 1))

    ##############################################################################
    ######################## Initialise Memory Objects: ##########################
    ##############################################################################

    # Profit - Value of immediate exercise:
    profit = numpy.zeros(state_variables.shape)

    # American option value of project, given that you can either delay or exercise:
    american_option_value = numpy.zeros(number_simulations)

    # Optimal period of exercise is the earliest time that exercise is triggered. If no exercise, an NA is returned:
    exercise_timing = numpy.full(shape=number_simulations, fill_value=numpy.nan)

    ## Payoff function:
    if call:
        profit_function = lambda payoff, strike_price: numpy.maximum(payoff - strike_price, 0)
    else:
        profit_function = lambda payoff, strike_price: numpy.maximum(strike_price - payoff, 0)

    ##############################################################################
    ###################### Begin LSM Simulation Algorithm: #######################
    ##############################################################################

    # Option maturity - the Immediate payoff of exercising:
    profit[-1, :] = profit_function(payoff[-1, :], strike_price)

    ## Would we exercise at option termination?
    exercise = profit[-1,] > 0

    # Receive immediate profit if exercising:
    american_option_value[exercise] = profit[-1, exercise]

    # Was the option exercised?
    exercise_timing[exercise] = t

    ## American Options hold value in waiting:
    ## Backwards induction begin:
    t = termination_period -1 
    for t in range(termination_period - 1, 0, -1):
        # t is representative of the index for time, but in reality the actual time period you're in is (t-1).

        ## Forward insight (high bias) - the immediate payoff of exercise:
        profit[t, :] = profit_function(payoff[-1, :], strike_price)
        profit_t = profit[t, :]

        # We only consider the exercise / delay exercise decision for price paths that are in the money (ie. profit from immediate exercise > 0):
        state_variables_t = state_variables[t, :]
        in_the_money_paths = profit_t > 0

        # Expected value of waiting to exercise - Continuation value:
        continuation_value = american_option_value * discount_rate

        # Use Least-Squares regression to introduce low bias, and compare expected value of waiting against the value of immediate exercise:
        if in_the_money_paths.any():
            continuation_value = estimate_continuation_value(
                profit_t=profit_t,
                in_the_money_paths=in_the_money_paths,
                continuation_value=continuation_value,
                state_variables_t=state_variables_t,
                t=t,
                orthogonal=orthogonal,
                degree=degree,
                cross_product=cross_product)

        # STEP THREE:
        # Dynamic programming:
        exercise = profit[t,] > continuation_value

        # Discount existing values if not exercising
        american_option_value[~exercise] = american_option_value[~exercise] * discount_rate
        # Receive immediate profit if exercising
        american_option_value[exercise] = profit[t, exercise]

        # Was the option exercised?
        exercise_timing[exercise] = t

        # Re-iterate.
    # End backwards induction.

    # Calculate project value -  discount payoffs
    option_values = numpy.repeat(0, number_simulations)

    # American option value - discounting payoffs back to time zero, averaging over all paths.
    exercised = ~numpy.isnan(exercise_timing)
    exercise_period = exercise_timing[exercised]
    option_values[exercised] = profit[number_periods * (in_the_money_paths-1) + exercise_period] * discount(nominal_interest_rate, exercise_period)

    # Calculate option value:
    option_price = numpy.mean(option_values)

    # Execise time:
    exercise_time = (exercise_period - 1) * time_step

    return numpy.mean(option_values)
    # return option_results(option_price, option_values, number_simulations, exercise_time, in_the_money_paths, time_step)
