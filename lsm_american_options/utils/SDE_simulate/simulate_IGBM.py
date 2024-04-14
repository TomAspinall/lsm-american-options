from numbers import Number
import numpy as np
from math import sqrt, log, exp


## 100 simulations of 1 year of monthly price paths:
n = 100
t = 1
reversion_rate = 1
sigma = 0.2
equilibrium = 100
S0 = 100
dt = 1/12

def simulate_IGBM(
        n: int, 
        t: Number, 
        reversion_rate: Number,
        equilibrium: Number,
        sigma: Number, 
        S0: Number, 
        dt: Number
        ) -> np.ndarray:

    ## Dimension 1:
    number_steps = t / dt
    from math import ceil
    number_steps_total = ceil(number_steps)

    ## Dimension 2:
    if n % 2 == 0:
        number_simulations = n
    else:
        number_simulations = n + 1
    number_loops = int(number_simulations / 2)


    ## shock:
    shock = (np.random.normal(scale=sigma * sqrt(dt), size = number_loops * number_steps_total)).reshape((number_steps_total, number_loops))

    output = np.ndarray((number_steps_total, number_simulations))

    ## Initial values;
    output[0,:] = log(S0)

    adjusted_risk = 0.5 * (sigma ** 2)
    simulated_value_columns = np.arange(0,number_simulations,2)
    antithetic_value_columns = simulated_value_columns + 1
    for t in range(number_steps_total):
        output_exp_t = np.exp(output[t])
        ## Log-distribution:
        
        ## Mean-reverting drift:
        drift = (reversion_rate * ((equilibrium - output_exp_t) / output_exp_t) - adjusted_risk) * dt

        ## Simulated Values:
        output[t, simulated_value_columns] += drift[simulated_value_columns] + shock[t,:]
        
        ## Antithetic Values:
        output[t, antithetic_value_columns] += drift[antithetic_value_columns] + shock[t,:]

    ## Return output:
    return np.exp(output)

# simulate_IGBM(n, t, reversion_rate, equilibrium, sigma, S0, dt)



