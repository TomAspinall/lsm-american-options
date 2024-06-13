from math import sqrt
from statistics import mean, variance
import pandas as pd
import numpy as np

class AmericanOption():
    
    def __init__(
            self, 
            american_option_value,
            number_simulations, 
            exercise_time, 
            number_periods,
            time_step,
            call_option
            ):

        self.call_option = call_option

        # American option value - discounting payoffs back to time zero, averaging over all paths.
        exercised = ~np.isnan(exercise_time)
        exercise_period = exercise_time[exercised].astype(int)

        # american_option_value[exercised] = profit[exercise_period,exercised] * np.exp(- nominal_interest_rate * exercise_period)

        # Option value:
        self.option_price = np.mean(american_option_value)

        # Execise time:
        exercise_time = exercise_period * time_step

        # Option value standard error:
        self.standard_error = sqrt(variance(american_option_value) / number_simulations)

        # Expected exercise time
        self.expected_exercise_years = mean(exercise_time)

        # Exercise probability:
        self.exercise_probability = len(exercise_period) / number_simulations

        # Exercise timings and cumulative exercise probability:
        unique, counts = np.unique(exercise_time, return_counts=True)
        all_time_steps = np.arange(0, time_step * number_periods, time_step)
        exercise_probability = pd.DataFrame(index=all_time_steps)
        exercise_probability.loc[unique, 'counts'] = counts

        self.exercise_timings = exercise_probability
        self.exercise_probability_cumulative = np.cumsum(exercise_probability['counts'].fillna(0) / number_simulations)

        def __repr__(self):
            if self.call_option:
                option = "call"
            else:
                option = "put"
            return f"American {option} option: {self.american_option_value:.2f} ({self.standard_error})"
        
        def __str__(self):
            if self.call_option:
                option = "call"
            else:
                option = "put"
            return f"American {option} option: {self.american_option_value:.2f} ({self.standard_error})"
