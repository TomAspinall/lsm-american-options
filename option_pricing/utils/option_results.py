from math import sqrt
from statistics import mean, variance
import pandas as pd
import numpy as np

## TODO: np.var vs statistics.variance (div n vs. n - 1)

class AmericanOption():
    
    def __init__(
            self, 
            american_option_value,
            number_simulations, 
            exercise_timings, 
            number_periods,
            time_step,
            call_option
            ):

        self.call_option = call_option

        # American option value - discounting payoffs back to time zero, averaging over all paths.
        exercised = ~np.isnan(exercise_timings)
        exercise_period = exercise_timings[exercised].astype(int)

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

class RealOption():
    def __init__(
            self, 
            profit,
            real_option_value,
            exercise_timings, 
            net_cash_flow,
            capital_expenditure,
            construction_periods,
            number_simulations, 
            number_periods,
            time_step
            ):

        # Investment Values:
        ## Real Option Value (ROV):
        self.ROV = np.mean(real_option_value)
        ## Net Present Value (NPV):
        self.NPV = np.mean(profit[0,])
        ## Waiting Option Value (WOV):
        self.WOV = self.ROV - self.NPV

        # Simulation Standard Errors:
        from math import sqrt
        self.ROV_SE = sqrt(variance(real_option_value) / number_simulations)
        self.NPV_SE = sqrt(variance(profit[0,]) / number_simulations)
        self.WOV_SE = sqrt(variance(real_option_value - profit[0,]) / number_simulations)

        # American option value - discounting payoffs back to time zero, averaging over all paths.
        exercised = ~np.isnan(exercise_timings)
        exercise_period = exercise_timings[exercised].astype(int)

        # Execise time:
        exercise_time = exercise_period * time_step

        # Expected exercise time
        self.expected_exercise_years = mean(exercise_time) if len(exercise_time) > 0 else None

        # Exercise probability:
        self.exercise_probability = len(exercise_period) / number_simulations

        # Exercise timings and cumulative exercise probability:
        unique, counts = np.unique(exercise_time, return_counts=True)
        all_time_steps = np.arange(0, time_step * number_periods, time_step)
        exercise_probability = pd.DataFrame(index=all_time_steps)
        exercise_probability.loc[unique, 'counts'] = counts

        self.exercise_probability_cumulative = np.cumsum(exercise_probability['counts'].fillna(0) / number_simulations)
        
        ### Expected Payback Period (Conditional on investment exercised):
        # Time point at which net cash flows are accrued:
        benefits_accrued = exercise_period + construction_periods
        
        invested_NCF = net_cash_flow[:,exercised]
        ## TODO: Optimise:
        ## number_periods << number_simulations, row operations is preferable here:
        for i in range(len(invested_NCF)):
            invested_NCF[i, i < benefits_accrued] = 0
        ## When (if ever) do accrued benefits of investment exceed capital investment?
        
        ## TODO: Vectorised CAPEX support:
        payback_achieved = np.argmax(np.cumsum(invested_NCF, axis=0) > capital_expenditure, axis=0)
        ## Final check - are the 0's real?
        ## Payback time from investment to making back capital invesment:
        payback = (payback_achieved - exercise_period) * time_step
        
        ## If you invest, and it does make back the initial capital expenditure, it's expected to take this long:
        payback_boolean = payback >= 0
        payed_back = payback_boolean.sum()
        if payed_back > 0:
            self.expected_payback = mean(payback[payback_boolean])
            ## Corresponding standard error:
            self.expected_payback_SE = variance(payback[payback_boolean])
        else:
            self.expected_payback = np.nan
            self.expected_payback_SE = np.nan
        ## If you invest, you have this probability of making back the initial capital expenditure:
        self.expected_payback_probability = payed_back / len(payback)

        ## Validation:
        # real_option_value[exercised] = profit[exercise_period,exercised] * np.exp(- nominal_interest_rate * exercise_period)

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
