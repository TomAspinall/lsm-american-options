from math import exp

# Helper:

def discount(interest_rate, time_period=1): 
    return exp(-interest_rate*time_period)
