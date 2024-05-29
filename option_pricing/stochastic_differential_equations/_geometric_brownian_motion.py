from numbers import Number
import numpy as np
from math import sqrt, log, ceil

# TODO: t // dt, n % 2 != 0:

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

    ## natural log initial spot prices:
    ln_S0 = log(S0)

    # Drift:
    drift = (mu - (0.5 * sigma**2)) * dt
    # Cumulative Drift per time point:
    drift_t = np.cumsum(np.repeat(drift, number_steps))
    # Shock:
    shock = np.random.normal(loc = 0, scale=sigma, size=number_loops *
                             number_steps).reshape((number_loops, number_steps)) * sqrt(dt)

    shock_cumulative = np.cumsum(shock, axis=1)

    ## Initial values:
    initial = np.repeat(ln_S0, number_simulations).reshape(number_simulations, 1)

    ## Output array - cols by rows:
    output = np.zeros((number_simulations, number_steps))

    ## Thetic values:
    output[:number_loops,:] = ln_S0 + np.add(drift_t, shock_cumulative)
    ## Antithetic values:
    output[(number_loops):(number_simulations+1),:] = ln_S0 + np.add(drift_t, -shock_cumulative)

    ## Final output:
    ## TODO: Remove transpose:
    return np.exp(np.concatenate([initial, output], axis=1)).transpose()
