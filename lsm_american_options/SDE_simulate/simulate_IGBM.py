from numbers import Number
import numpy as np
from math import sqrt, log, ceil


def simulate_IGBM(
        n: int,
        t: Number,
        reversion_rate: Number,
        equilibrium: Number,
        sigma: Number,
        S0: Number,
        dt: Number
) -> np.ndarray:

    # Constant:
    adjusted_risk = 0.5 * (sigma ** 2)

    # Dimension 1:
    number_steps = t / dt
    number_steps_total = ceil(number_steps)

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
    shock = (np.random.normal(scale=sigma * sqrt(dt), size=number_loops *
             number_steps_total)).reshape((number_steps_total, number_loops))

    # Output array:
    output = np.ndarray((number_steps_total, number_simulations))

    # Initial values;
    output[0] = log(S0)

    # Begin Monte Carlo simulation:
    for t in range(number_steps_total - 1):

        # Log-distribution:
        output_exp_t = np.exp(output[t])

        # Mean-reverting drift:
        drift = (reversion_rate * ((equilibrium - output_exp_t) /
                 output_exp_t) - adjusted_risk) * dt

        # Simulated Values:
        output[t + 1, simulated_value_columns] = output[t,
                                                        simulated_value_columns] + drift[simulated_value_columns] + shock[t]

        # Antithetic Values:
        output[t + 1, antithetic_value_columns] = output[t,
                                                         antithetic_value_columns] + drift[antithetic_value_columns] - shock[t]

    # Return output:
    return np.exp(output[:, :n])
