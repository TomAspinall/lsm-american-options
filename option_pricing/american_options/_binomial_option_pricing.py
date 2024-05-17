from math import exp, sqrt, ceil
import numpy as np

## Discount Factor:
from .._utils._discount import discount

def discount(interest_rate, time_period=1): 
    return exp(-interest_rate*time_period)


# Functions for probabilities & tree prices:

# r = 0.05
# delta_t = 1
# stock_price = 18.01
# strike_price = 20
# n = 40
# call_option = True
# mu = 0.06
# sigma = 0.2
# S0 = 38
## GBM Parameters:
# alpha = 0.0408
# risk_premium = 0.01
# sigma = 0.4683

## TODO: Payoff for American option call / put
## 2024-04-30: Needs more information

def binomial_option_pricing_model(
    r,
    delta_t,
    sigma,
    stock_price,
    strike_price,
    n,
    call_option = True
):

    #Drift value:
    drift = alpha - 0.5 * sigma - risk_premium


    ## Discount rate between discrete steps:
    discount_rate = discount(r, delta_t)

    ## Number of Discrete time steps:
    number_periods = n / delta_t
    ## Total Periods evaluated (int):
    total_periods = ceil(number_periods)

    ## Associated matrices shape:
    out_shape = (total_periods, total_periods)

    ## Risk Neutral Probabilites:
    prob_up   = exp(sigma * sqrt(delta_t))
    prob_down =  1 / prob_up
    
    ## Discount likelihood according to probabilities:
    probability = (exp((r)*delta_t)-prob_down)/(prob_up-prob_down)

    ## Sparse / Diagonal tree of probabilities:
    potential_stock_prices = np.zeros(out_shape)

    ## Calculate Cumulative Binomial Tree probabilities:
    for index, time in zip(range(total_periods), range(1,total_periods+1)):
        cols = np.arange(start=1, stop=time+1)

        down = prob_down ** (time - cols)
        up = prob_up ** (cols - 1)
        
        potential_stock_prices[index, :time] = np.maximum(stock_price * up * down, 0)

    ## Payoff / Decision tree:
    option_tree = np.zeros(out_shape)
    optimal_decisions = np.zeros(out_shape)

    ## Payoff calculation:
    if call_option:
        payoff = lambda stock_price, option_payoff: max(stock_price - strike_price, option_payoff)
    else:
        payoff = lambda stock_price, option_payoff: max(strike_price - stock_price, option_payoff)

    ## Begin backwards induction:
    for index, time in zip(range(total_periods - 1, -1, -1), range(total_periods, 0, -1)):
        break
        
        ## Discounted payoff of immediate exercise:
        potential_stock_prices[index, :time] * discount(r, delta_t * index)



        ## Exercise Payoff:
        exercise_payoff = payoff()
        print(index)
        break

    # option_tree[-1,:] = 0