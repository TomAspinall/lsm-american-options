from numbers import Number
import numpy as np
from math import sqrt, log

n = 100
t = 1
mu = 0.05
sigma = 0.2
S0 = 100
dt = 1/12


def geometric_brownian_motion(
        n: int,
        t: Number,
        mu: Number,
        sigma: Number,
        S0: Number,
        dt: Number
) -> np.ndarray:

    # Dimension 1:
    number_steps = t / dt
    from math import ceil
    number_steps_total = ceil(number_steps)

    # Dimension 2:
    if n % 2 == 0:
        number_simulations = n
    else:
        number_simulations = n + 1
    number_loops = int(number_simulations / 2)

    # Drift:
    drift = (mu - 0.5 * (sigma**2)) * dt
    # Drift per time point:
    # drift_t = np.cumsum(np.repeat(drift, number_loops))
    drift_t = np.repeat(drift, number_loops)
    # Applied to each simulation:
    # Shock:
    shock = np.random.normal(scale=sigma * sqrt(dt), size=number_loops *
                             number_steps_total).reshape((number_steps_total, number_loops))

    # Simulated values:
    values = np.cumsum(drift_t) + np.cumsum(shock, axis=1)
    # Antithetic values:
    antithetic_values = np.cumsum(drift_t) - np.cumsum(shock, axis=1)

    output = np.ndarray((number_steps_total + 1, number_simulations))
    # Initial value:
    output[0, :] = log(S0)
    # Simulated values:
    output[1:, :] = log(
        S0) + np.concatenate([values, antithetic_values], axis=1)

    # values = log(S0) + np.cumsum(drift_t + shock, axis = 1)
    # np.cumsum(drift_t + shock, axis = 1).round(2)
    # antithetic_values = log(S0) + np.cumsum(drift_t - shock, axis = 1)
    # output = np.concatenate([values, antithetic_values], axis=1)

    return np.exp(output)


# Why do the antithetic values go haywire after a certain point?
# hi = GBM_simulate(
#     n=100000,
#     t=40,
#     mu=0.05,
#     sigma=0.2,
#     S0=100,
#     dt=1/12
# )