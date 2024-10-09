from numbers import Number
from typing import Optional

import numpy as np

# Requirements:
from ..utils.continuation_value import estimate_continuation_value
from ..utils.discount import discount
from ..utils.option_results import AmericanOption

# __name__ = "option_pricing.american_options.monte_carlo_simulation"


def monte_carlo_simulation(state_variables: np.ndarray,
                           payoff: np.ndarray,
                           strike_price: Number,
                           time_step: Number,
                           risk_free_rate: Number,
                           call_option: Optional[bool] = True,
                           orthogonal: Optional[str] = "Power",
                           degree: Optional[int] = 2,
                           cross_product: Optional[bool] = True,
                           ):

    # State variables must be coerced as a 3-d array:
    if state_variables.ndim < 3:
        state_variables = state_variables.reshape(state_variables.shape + (1,))

    # Const:

    # length rolumns: # simulations
    # length rows:    # discrete time periods
    # length slices:  # underlying state variables
    # number_periods, number_simulations, number_state_variables = state_variables.shape
    number_periods, number_simulations, number_state_variables = state_variables.shape

    # Nominal interest rate:
    nominal_interest_rate = risk_free_rate * time_step
    # Corresponding discount:
    discount_rate = discount(nominal_interest_rate)

    # Time period of backwards induction:
    termination_period = number_periods - 1

    # Assertions:
    # assert type(K) is Number and not len(K) == number_simulations, "length of object 'K' does not equal 1 or number of columns of 'state_variables'!"
    assert not np.isnan(state_variables).any(
    ), "NA's cannot be specified within 'state_variables'"
    assert number_periods == payoff.shape[0] and number_simulations == payoff.shape[
        1], "The first 2 dimensions of 'state_variables' must match the dimensions of 'payoff'"

    ##############################################################################
    ################### Calculate Immediate Payoff (High Bias): ##################
    ##############################################################################

    # Payoff function:
    if call_option:
        def profit_function(payoff, strike_price): return np.maximum(
            payoff - strike_price, 0)
    else:
        def profit_function(payoff, strike_price): return np.maximum(
            strike_price - payoff, 0)

    # Forward insight (high bias) - the immediate payoff of exercise at any time point and simulated payoff path:
    profit = profit_function(payoff, strike_price)

    ##############################################################################
    ###################### Begin LSM Simulation Algorithm: #######################
    ##############################################################################

    # American option value of project, given that you can either delay or exercise:
    american_option_value = np.zeros(shape=number_simulations)

    # Optimal period of exercise is the earliest time that exercise is triggered. If no exercise, an NA is returned:
    exercise_timings = np.full(shape=number_simulations, fill_value=np.nan)

    # Would we exercise at option termination?
    exercise = profit[-1,] > 0

    # Receive immediate profit if exercising:
    american_option_value[exercise] = profit[-1, exercise]

    # Was the option exercised?
    exercise_timings[exercise] = termination_period

    # American Options hold value in waiting:
    # Backwards induction begin:
    # t = termination_period - 1
    for t in range(termination_period - 1, -1, -1):

        # Immediate payoff of exercise:
        profit_t = profit[t, :]

        # We only consider the exercise / delay exercise decision for price paths that are in the money (ie. profit from immediate exercise > 0):
        state_variables_t = state_variables[t, :, :]
        in_the_money_paths = profit_t > 0

        # Expected value of waiting to exercise - Continuation value:
        continuation_value = american_option_value * discount_rate

        # Least-Squares regression (low bias) - compare expected value of waiting against the value of immediate exercise:
        if in_the_money_paths.any():
            continuation_value = estimate_continuation_value(
                in_the_money_paths=in_the_money_paths,
                continuation_value=continuation_value,
                state_variables_t=state_variables_t,
                orthogonal=orthogonal,
                degree=degree,
                cross_product=cross_product)

        # Dynamic programming:
        exercise = profit_t > continuation_value

        # Discount existing values if not exercising
        american_option_value[~exercise] = american_option_value[~exercise] * discount_rate
        # Receive immediate profit if exercising
        american_option_value[exercise] = profit_t[exercise]

        # Was the option exercised?
        exercise_timings[exercise] = t

        # Re-iterate.
    # End backwards induction.

    # TODO: One more required?
    american_option_value = american_option_value * discount_rate

    # Evaluate outputs:
    return AmericanOption(
        american_option_value=american_option_value,
        number_simulations=number_simulations,
        exercise_timings=exercise_timings,
        number_periods=number_periods,
        time_step=time_step,
        call_option=call_option
    )
