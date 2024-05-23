
from math import exp, sqrt, ceil
import numpy as np

## Discount Factor:
from .._utils._discount import discount

def discount(interest_rate, time_period=1): 
    return exp(-interest_rate*time_period)

# Functions for probabilities & tree prices:

r = 0.05
strike_price = 20

delta_t = 1
stock_price = 18.01
n = 40
call_option = False
mu = 0.06
sigma = 0.2
S0 = 38
# GBM Parameters:
# alpha = 0.0408
alpha = 0
risk_premium = 0
sigma = 0.4683


def binomial_option_pricing_model(
    r,
    delta_t,
    sigma,
    stock_price,
    strike_price,
    n,
    call_option = True
):

    ## Discount / premium rate between discrete steps:
    discount_rate = discount(r, delta_t)
    premium_rate = discount(-r, delta_t)

    ## Number of Discrete time steps:
    number_periods = n / delta_t
    ## Total Periods evaluated (int):
    total_periods = ceil(number_periods)

    ## Associated matrices shape:
    out_shape = (total_periods, total_periods)

    ## Risk Neutral Probabilites:
    prob_up   = exp(sigma * sqrt(delta_t))
    prob_down =  exp(-sigma * sqrt(delta_t))
    
    ## Discount likelihood according to probabilities:
    probability = (premium_rate-prob_down)/(prob_up-prob_down) if prob_up-prob_down != 0 else 1
    q_probability = 1 - probability

    ## Sparse / Diagonal tree of probabilities:
    potential_stock_prices = np.zeros(out_shape)

    ## Calculate Cumulative Binomial Tree probabilities:
    for index, time in zip(range(total_periods), range(1,total_periods+1)):
        cols = np.arange(start=1, stop=time+1)

        ## Probabilities:
        down = prob_down ** (time - cols)
        up = prob_up ** (cols - 1)

        potential_stock_prices[index, :time] = up * down
    ## Stock Price Tree:
    potential_stock_prices *= stock_price

    # terminal_stock_prices = stock_price * prob_down ** (total_periods - cols) * prob_up ** (cols - 1)


    ## Safety:
    # potential_stock_prices = np.maximum(potential_stock_prices, 0)

    ## Payoff / Decision tree:
    option_tree = np.zeros(out_shape)

    ## Payoff calculation:
    if call_option:
        option_tree[-1, :] = np.maximum(potential_stock_prices[-1, :] - strike_price, 0)
    else:
        option_tree[-1, :] = np.maximum(strike_price - potential_stock_prices[-1, :], 0)

    ## Recursive payoff calculation:
    for i in range(len(option_tree) - 1, 0, -1):
        option_tree[i-1, :i] = discount_rate * (probability * (option_tree[i, :i]) + (q_probability * option_tree[i, 1:(i+1)]))

    ## Final calculation:
    return discount_rate * (probability * (option_tree[1, 0]) + (q_probability * option_tree[1, 1]))