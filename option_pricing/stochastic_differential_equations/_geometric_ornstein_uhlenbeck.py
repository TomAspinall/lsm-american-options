from numbers import Number
import numpy as np
from math import sqrt, log, ceil

def geometric_ornstein_uhlenbeck(
        n: int,
        t: Number,
        reversion_rate: Number,
        sigma: Number,
        equilibrium: Number,
        risk_premium: Number,
        S0: Number,
        dt: Number
) -> np.ndarray:
    """
    Simulate the geometric Ornstein-Uhlenbeck (GOU) stochastic process through Monte Carlo simulation and antithetic variates.

    The geometric Ornstein-Uhlenbeck process is a member of the general affine class of stochastic process. The Ornstein-Uhlenbeck process is a 
    Gaussian process, a Markov process, is temporally homogeneous and exhibits mean-reverting behaviour.    
    """

    # Constant:
    risk_equilibrium = risk_premium / reversion_rate

    # Dimension 1:
    number_steps = t / dt
    number_simulated_steps = ceil(number_steps)
    # Include initial value:
    number_steps_total = number_simulated_steps + 1

    # Dimension 2:
    if n % 2 == 0:
        number_simulations = n
    else:
        number_simulations = n + 1

    # Even number of simulations conducted regardless:
    number_loops = int(number_simulations / 2)

    # Thetic and antithetic columns:
    simulated_value_columns = np.arange(0, number_simulations, 2)
    antithetic_value_columns = simulated_value_columns + 1

    # shock:
    shock = (np.random.normal(scale=sigma * sqrt(dt), size=number_loops * number_simulated_steps)).reshape((number_simulated_steps, number_loops))

    # Output array:
    output = np.zeros((number_steps_total, number_simulations))

    # Initial values;
    output[0] = log(S0) - log(equilibrium)

    # Begin Monte Carlo simulation:
    t = 0
    for t in range(1, number_steps_total):

        drift = reversion_rate * (risk_equilibrium - output[t]) * dt

        # Simulated Values:
        output[t, simulated_value_columns] = output[t - 1,
                                                        simulated_value_columns] + drift[simulated_value_columns] + shock[t]

        # Antithetic Values:
        output[t, antithetic_value_columns] = output[t - 1,
                                                         antithetic_value_columns] + drift[antithetic_value_columns] - shock[t]

    # Return output:
    return np.exp(output[:, :n] + log(equilibrium))
