import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt 

def optimize_portfolio(returns):
    expected_returns = returns.mean()
    cov_matrix = returns.cov()

    # Portfolio Optimization Function
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    # Constraints and bounds
    num_assets = len(expected_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    opt_results = minimize(
        portfolio_volatility, 
        initial_weights, 
        args=(cov_matrix,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return opt_results.x

def dynamic_portfolio(returns, rebalance_period=10, risk_free_rate=0.01):
    num_periods = len(returns)
    portfolio_values = [1.0]  # Starting portfolio value

    for i in range(0, num_periods, rebalance_period):
        # Use returns up to the current period for optimization
        if i == 0:
            continue
        rebalancing_returns = returns.iloc[:i]
        optimal_weights = optimize_portfolio(rebalancing_returns)

        # Calculate portfolio return for the next period
        period_returns = returns.iloc[i:i + rebalance_period]
        weighted_returns = np.dot(period_returns, optimal_weights)
        period_portfolio_return = (1 + weighted_returns).prod()
        portfolio_values.append(portfolio_values[-1] * period_portfolio_return)
    
    return portfolio_values

def print_portfolio_values(portfolio_values):
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, label='Portfolio Value')
    plt.title('Dynamically Optimized Portfolio Value')
    plt.xlabel('Rebalancing Periods')
    plt.ylabel('Portfolio Value')
    plt.show()

# Parameters
rebalance_period=3
risk_free_rate=0.01

# Import Prices
tickers = [
    'AAPL',  # Apple Inc.
    'MSFT',  # Microsoft Corporation
    'NVDA',  # NVIDIA Corporation
    'AMZN',  # Amazon.com Inc.
    'GOOGL', # Alphabet Inc. Class A
    'GOOG',  # Alphabet Inc. Class C
    'META',  # Meta Platforms Inc.
    'TSLA',  # Tesla Inc.
    'AVGO',  # Broadcom Inc.
    'PEP',   # PepsiCo Inc.
    'ADBE',  # Adobe Inc.
    'COST',  # Costco Wholesale Corp.
    'CSCO',  # Cisco Systems Inc.
    'CMCSA', # Comcast Corp.
    'NFLX',  # Netflix Inc.
    'TXN',   # Texas Instruments Inc.
    'INTC',  # Intel Corp.
    'AMGN',  # Amgen Inc.
    'QCOM',  # Qualcomm Inc.
    'HON',   # Honeywell International Inc.
]
data = yf.download(
    tickers,
    start="2024-01-01",
    end="2025-05-05",
    progress=False
)
# Close prices
prices = data["Close"][tickers].dropna()
# Log returns
returns = np.log(prices / prices.shift(1))


portfolio_values = dynamic_portfolio(returns)
print_portfolio_values(portfolio_values)
