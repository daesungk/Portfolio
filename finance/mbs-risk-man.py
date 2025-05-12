import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Interest rate paths
def interest_rate_paths(r0, mean_reversion, volatility, long_term_rate, num_years, steps_per_year, num_simulations, interest_model):
    num_steps = num_years * steps_per_year
    rates = np.zeros((num_simulations, num_steps))
    rates[:, 0] = r0
    dt = 1.0 / steps_per_year

    for i in range(1, num_steps):
        if interest_model == 'Hull-White':
            rates[:, i] = rates[:, i - 1] + mean_reversion * (long_term_rate - rates[:, i - 1]) * dt \
                          + volatility * np.sqrt(dt) * np.random.normal(size=num_simulations)
        elif interest_model == 'CIR':
            rates[:, i] = rates[:, i - 1] + mean_reversion * (long_term_rate - rates[:, i - 1]) * dt \
                          + volatility * np.sqrt(np.maximum(rates[:, i - 1], 0)) * np.sqrt(dt) * np.random.normal(size=num_simulations)
        elif interest_model == 'Vasicek':
            rates[:, i] = rates[:, i - 1] + mean_reversion * (long_term_rate - rates[:, i - 1]) * dt \
                          + volatility * np.sqrt(dt) * np.random.normal(size=num_simulations)

    return rates

# Prepayment Rates
def psa_rates(percentage, num_steps, num_simulations):
    per = percentage / 100.0
    base_psa = np.array([min(0.002 * month, 0.06) for month in range(1, num_steps + 1)])
    psa_matrix = np.tile(base_psa * per, (num_simulations, 1))
    return psa_matrix

def cpr_rates(base_cpr, sensitivity, rates):
    return base_cpr * np.exp(-sensitivity * rates)

def logistic_rates(intercept, coef, rates):
    odds = np.exp(intercept + coef * rates)
    return odds / (1 + odds)

# Parameters
num_simulations = 10000
num_years = 30
steps_per_year = 12

mean_reversion = 0.03
volatility = 0.01
r0 = 0.03
long_term_rate = 0.03

# Determine Models
interest_model = 'CIR'  # 'Hull-White', 'CIR', 'Vasicek'
prepayment_model = 'PSA'  # 'PSA', 'CPR', 'Logistic'

# Generate interest rate paths
num_steps = num_years * steps_per_year
rates = interest_rate_paths(r0, mean_reversion, volatility, long_term_rate, num_years, steps_per_year, num_simulations, interest_model)

# Parameters for prepayment
percentage = 150  # PSA 100, 150, or 200
base_cpr = 0.03   # CPR
sensitivity = 1.2 # CPR
intercept = -2.0  # Logistic
coef = 3.0        # Logistic

# Generate prepayment rates
if prepayment_model == 'PSA':
    prepayment_rates = psa_rates(percentage, num_steps, num_simulations)
elif prepayment_model == 'CPR':
    prepayment_rates = cpr_rates(base_cpr, sensitivity, rates)
elif prepayment_model == 'Logistic':
    prepayment_rates = logistic_rates(intercept, coef, rates)

# Simulate Cash Flows
principal = 1000000

def mbs_cashflow(principal, prepayment_rates, rates):
    cash_flows = np.zeros_like(prepayment_rates)
    remaining_principal = np.full((prepayment_rates.shape[0],), float(principal))

    for i in range(prepayment_rates.shape[1]):
        interest_payment = remaining_principal * 0.04 / steps_per_year
        prepayment_amount = remaining_principal * prepayment_rates[:, i]
        cash_flows[:, i] = interest_payment + prepayment_amount
        remaining_principal -= prepayment_amount

    return cash_flows

# Calculate Cash Flows
cash_flows = mbs_cashflow(principal, prepayment_rates, rates)

# Calculate VaR and CVaR
present_values = cash_flows.sum(axis=1) / (1 + rates[:, -1])**num_years
var_95 = np.percentile(present_values, 5)
cvar_95 = present_values[present_values <= var_95].mean()

print(f"Mean Present Value: {present_values.mean()}")
print(f"VaR (95%): {var_95}")
print(f"CVaR (95%): {cvar_95}")
