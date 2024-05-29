from math import exp, sqrt, ceil
import numpy as np

## Discount Factor:
from .._utils._discount import discount

# def discount(interest_rate, time_period=1): 
#     return exp(-interest_rate*time_period)

def binomial_option_pricing_model(
    risk_free_rate,
    time_step,
    sigma,
    stock_price,
    strike_price,
    n,
    call_option = True
):

    ## Discount / premium rate between discrete steps:
    discount_rate = discount(risk_free_rate, time_step)
    premium_rate = discount(-risk_free_rate, time_step)

    ## Number of Discrete time steps:
    number_periods = n / time_step
    ## Total Periods evaluated (int):
    total_periods = ceil(number_periods)

    ## Risk Neutral Probabilites:
    prob_up   = exp(sigma * sqrt(time_step))
    prob_down =  exp(-sigma * sqrt(time_step))
    
    ## Length of output:
    cols = np.arange(start=1, stop=total_periods+1)

    ## Prices at option maturity:
    terminal_stock_prices = stock_price * prob_down ** (total_periods - cols) * prob_up ** (cols - 1)

    ## Payoff calculation:
    if call_option:
        option_tree = np.maximum(terminal_stock_prices - strike_price, 0)
    else:
        option_tree = np.maximum(strike_price - terminal_stock_prices, 0)

    ## Discount likelihood according to probabilities:
    probability = (premium_rate-prob_down)/(prob_up-prob_down) if prob_up-prob_down != 0 else 1
    q_probability = 1 - probability

    ## Recursive option pricing accoriding to risk-neutral probabilities of (up) vs (down):
    i = 0
    for i in range(total_periods -1 , 0, -1):
        option_tree[:i] = discount_rate * (probability * option_tree[:i] + q_probability * option_tree[1:(i+1)])
    return option_tree[i]