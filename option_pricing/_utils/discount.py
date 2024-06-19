from math import exp
import numpy as np
from numbers import Number
from typing import Optional, Union

# Helper:
def discount(
        interest_rate: Number, 
        time_period: Number=1): 
    return exp(-interest_rate*time_period)

def discount_array(
        interest_rate: Number, 
        time_periods: np.array): 
    return np.exp(-interest_rate*time_periods)
