from numbers import Number
import numpy as np
from math import sqrt, log, ceil

# n = 100
# t = 40
# mu = 0.05
# sigma = 0.001
# S0 = 100
# dt = 1/12


def geometric_brownian_motion(
        n: int,
        t: Number,
        mu: Number,
        sigma: Number,
        S0: Number,
        dt: Number
) -> np.ndarray:

    # Dimension 1:
    number_steps = ceil(t / dt)
    rounded_up = divmod(t, dt)[1] > 1e-5
    if rounded_up:
        number_steps += 1

    # Dimension 2:
    if n % 2 == 0:
        number_simulations = n
    else:
        number_simulations = n + 1
    number_loops = int(number_simulations / 2)

    # Drift:
    drift = (mu - 0.5 * (sigma**2)) * dt
    # Cumulative Drift per time point:
    drift_t = np.cumsum(np.repeat(drift, number_steps))
    # Shock:
    shock = np.random.normal(scale=sigma * sqrt(dt), size=number_loops *
                             number_steps).reshape((number_steps, number_loops))
    shock_cumulative = np.cumsum(shock, axis=1)

    ## Total output steps includes today:
    number_steps_output = number_steps + 1

    ## Output array:
    output = np.zeros((number_steps_output, number_loops))
    antithetic_output = np.zeros((number_steps_output, number_loops))

    ## Initial value:
    output[0, :] = log(S0)
    antithetic_output[0, :] = log(S0)

    # Simulated values:
    for i in range(number_loops):
        output[1:,i] = log(S0) + drift_t + shock_cumulative[:,i]
        antithetic_output[1:,i] = log(S0) + drift_t - shock_cumulative[:,i]

    ## Output:
    return np.exp(np.concatenate([output, antithetic_output], axis=1))

# hi = geometric_brownian_motion(
#         n,
#         t,
#         mu,
#         sigma,
#         S0,
#         dt
# )

# hi.shape
# hi.max()