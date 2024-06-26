

import numpy as np
from numbers import Number
from typing import Optional, Union

__name__ = 'option_pricing.real_options.monte_carlo_simulation'

# Requirements:
from .._utils.continuation_value import estimate_continuation_value
from .._utils.discount import (
        discount,
        discount_array
        )
from .._utils.option_results import RealOption

from ..stochastic_differential_equations.geometric_brownian_motion import geometric_brownian_motion as GBM

## Step 1 - Simulate asset prices:
# state_variables = GBM(
#     n = 100,
#     t = 10,
#     mu = 0.05,
#     sigma = 0.2,
#     S0 = 100,
#     time_step= 1/2,
#     testing=False
# )

# state_variables
# net_cash_flow = state_variables - 100
# capital_expenditure = 2
# time_step = 1/2
# risk_free_rate = 0.05
# construction_periods = 0
# orthogonal = "Power"
# degree = 2
# cross_product = True

def monte_carlo_simulation(
        state_variables: np.ndarray,
        net_cash_flow: np.ndarray,
        capital_expenditure: Union[Number, np.array],
        time_step: Number,
        risk_free_rate: Number,
        construction_periods: int = 0,
        orthogonal: str = "Power",
        degree: int = 2,
        cross_product: bool = True,
        ):

    ## State variables must be coerced as a 3-d array:
    if state_variables.ndim < 3:
        state_variables = state_variables.reshape(state_variables.shape + (1,))

    ## Const:

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
    assert isinstance(construction_periods, int), "'construction_periods' must be of type 'int'"
    assert not np.isnan(state_variables).any(), "NA's cannot be specified within 'state_variables'"
    assert type(capital_expenditure) is not np.array or len(capital_expenditure) == len(net_cash_flow), "len(capital_expenditure) != len(net_cash_flow)"
    assert number_periods == net_cash_flow.shape[0] and number_simulations == net_cash_flow.shape[1], "The first 2 dimensions of 'state_variables' must match the dimensions of 'net_cash_flow'"

    ##############################################################################
    ################ Calculate Running Present Value (High Bias): ################
    ##############################################################################

    ## Array of subsequent discounting of cash flows:
    discount_arr = discount_array(nominal_interest_rate, np.array(range(number_periods)))
    
    ## Develop discount matrix:
    discount_matrix = np.eye(number_periods)
    diagonal = np.arange(0, number_periods)
    for i in diagonal[1:]:
        discount_matrix[diagonal[:-i] + i, diagonal[:-i]] = discount_arr[i]
    # Running Present Value (RPV) is the PV of all future net cash flows:
    running_present_value = (net_cash_flow.T @ discount_matrix).T
    ## TODO: RPV at terminal simulated price = 0? Assumption of if operating in this period or not?

    # Further discount the Running Present Value based upon waiting for construction to complete:
    if construction_periods > 0:
        running_present_value *= discount_arr[construction_periods]
    
    # Immediate profit (high bias) is the net present value (NVP) conditional on:
        # Expending Initial Capital Expenditure
        # Waiting the construction period
        # obtaining the RPV at the time point the project becomes operational.
    profit = np.zeros(shape=(number_periods, number_simulations))
    # Offset the running present value actually attained within the immediate profit by the number of periods to wait for construction to complete:
    profit[:(number_periods - construction_periods)] += running_present_value[construction_periods:number_periods]
    # Subtract the capital expenditure (which may be time varying):
    profit -= capital_expenditure

    ##############################################################################
    ###################### Begin LSM Simulation Algorithm: #######################
    ##############################################################################

    # Real option value of the project, given that the option to invest is exercised at any time point.
    real_option_value = np.zeros(shape=number_simulations)

    # Optimal period of exercise is the earliest time that exercise is triggered. If no exercise, an NA is returned:
    exercise_timings = np.full(shape=number_simulations, fill_value=np.nan)

    ## Would we exercise at option termination?
    exercise = profit[-1,] > 0

    # Receive immediate profit if exercising:
    real_option_value[exercise] = profit[-1, exercise]

    # Was the option exercised?
    exercise_timings[exercise] = termination_period

    ## American Options hold value in waiting:
    ## Backwards induction begin:
    for t in range(termination_period - 1, -1, -1):

        ## Immediate payoff of exercise:
        profit_t = profit[t, :]

        # We only consider the exercise / delay exercise decision for price paths that are in the money (ie. profit from immediate exercise > 0):
        state_variables_t = state_variables[t, :, :]
        in_the_money_paths = profit_t > 0

        # Expected value of waiting to exercise - Continuation value:
        continuation_value = real_option_value * discount_rate

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
        real_option_value[~exercise] = real_option_value[~exercise] * discount_rate
        # Receive immediate profit if exercising
        real_option_value[exercise] = profit_t[exercise]

        # Was the option exercised?
        exercise_timings[exercise] = t

        # Re-iterate.
    # End backwards induction.

    ## TODO: One more required inconsistency between real_options and american_options?
    # real_option_value = real_option_value * discount_rate

    ## Evaluate outputs:
    return RealOption(
        profit=profit,
        real_option_value=real_option_value,
        exercise_timings=exercise_timings, 
        net_cash_flow=net_cash_flow,
        capital_expenditure=capital_expenditure,
        construction_periods=construction_periods,
        number_simulations=number_simulations, 
        number_periods=number_periods,
        time_step=time_step
            )