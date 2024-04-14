from math import sqrt
from statistics import mean, variance
import pandas as pd
import numpy as np

class option_results():
    def __init__(self, option_price, option_values, number_simulations, exercise_time, in_the_money_paths, dt):

        ## Option value:
        self.option_price = option_price

        ## Option value standard error:
        self.standard_error = sqrt(variance(option_values) / number_simulations)

        ## Expected exercise time
        self.expected_exercise_years = mean(exercise_time)

        ## Exercise probability:
        self.exercise_probability = len(in_the_money_paths) / number_simulations

        ## Cumulative exercise probability:
        unique, counts = np.unique(exercise_time, return_counts=True)
        exercise_probs = pd.DataFrame(index = range(number_periods+1))
        exercise_probs.loc[unique, 'counts'] = counts
        cumulative_exercise_probability = np.cumsum(exercise_probs['counts'].fillna(0) / number_simulations)
        cumulative_exercise_probability.index *= dt
        self.cumulative_exercise_probability = cumulative_exercise_probability
        return self