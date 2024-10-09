# __name__ = 'option_pricing.european_options.monte_carlo_simulation'

# Imports:
from numbers import Number
from typing import Optional

import numpy

# Local Imports:
from ..utils.discount import discount


def monte_carlo_simulation(payoff: numpy.array,
                           strike_price: Number,
                           n: Number,
                           time_step: Number,
                           risk_free_rate: Number,
                           call_option: Optional[bool] = False,
                           ):

    # Nominal interest rate:
    nominal_interest_rate = risk_free_rate * time_step

    # Payoff calculation - option value at maturity:
    if call_option:
        profit = numpy.maximum(payoff - strike_price, 0)
    else:
        profit = numpy.maximum(strike_price - payoff, 0)

    # A European Option never holds a "continuation value" (expected value of waiting to exercise):
    # Discount payoffs:
    present_value = profit * discount(nominal_interest_rate, n)

    # Average simulated profit paths:
    return numpy.mean(present_value)
