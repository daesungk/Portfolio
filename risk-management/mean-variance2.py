import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import yfinance as yf

# Function to optimize portfolio for maximum mean return under volatility constraint
def optimize_portfolio_max_return(returns, max_volatility):
    expected_returns = returns.mean()
    cov_matrix = returns.cov()
    num_assets = len(expected_returns)

    # Objective: Maximize expected return (negative for minimization)
    def negative_return(weights):
        return -np.dot(weights, expected_returns)

    # Constraints: Weights sum to 1 and volatility constraint
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'ineq', 'fun': lambda x: max_volatility - np.sqrt(np.dot(x.T, np.dot(cov_matrix, x)))}
    ]
    
    # Bounds for weights (between 0 and 1)
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_weights = np.ones(num_assets) / num_assets

    # Optimization
    opt_results = minimize(
        negative_return, 
        initial_weights, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )
    
    return opt_results.x, -opt_results.fun 

# Dynamic Portfolio Optimization with Maximum Return under Volatility Constraint
def dynamic_portfolio_max_return(returns, max_volatility, rebalance_period, risk_free_rate):
    num_periods = len(returns)
    portfolio_values = [1.0]
    pnl = []

    for i in range(0, num_periods, rebalance_period):
        if i == 0:
            continue

        rebalancing_returns = returns.iloc[:i]
        optimal_weights, _ = optimize_portfolio_max_return(rebalancing_returns, max_volatility)
        period_returns = returns.iloc[i:i + rebalance_period]
        weighted_returns = np.dot(period_returns, optimal_weights)
        period_pnl = weighted_returns.sum()
        pnl.append(period_pnl)
        portfolio_values.append(portfolio_values[-1] * (1 + period_pnl))

    # Calculate overall Sharpe Ratio
    total_pnl = np.sum(pnl)
    avg_pnl = np.mean(pnl)
    pnl_volatility = np.std(pnl)
    sharpe_ratio = (avg_pnl - risk_free_rate) / pnl_volatility if pnl_volatility > 0 else float('inf')

    # Calculate drawdown
    portfolio_values = np.array(portfolio_values)
    drawdown = (portfolio_values / np.maximum.accumulate(portfolio_values)) - 1
    max_drawdown = drawdown.min()

    return portfolio_values, pnl, sharpe_ratio, max_drawdown

# Data Loading (Stock Prices)
tickers = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'GOOG', 
    'META', 'TSLA', 'AVGO', 'PEP', 'ADBE', 'COST', 
    'CSCO', 'CMCSA', 'NFLX', 'TXN', 'INTC', 'AMGN', 
    'QCOM', 'HON'
]

data = yf.download(
    tickers,
    start="2024-01-01",
    end="2025-05-05",
    progress=False
)

prices = data["Close"][tickers].dropna()
returns = np.log(prices / prices.shift(1)).dropna()

# Parameters
rebalance_period = 3
risk_free_rate = 0.01
max_volatility = 0.02  # Example: 2% maximum volatility

# Running the Dynamic Portfolio Optimization
portfolio_values, pnl, sharpe_ratio, max_drawdown = dynamic_portfolio_max_return(
    returns, max_volatility, rebalance_period, risk_free_rate
)

# Displaying Results
print("Final Portfolio Value:", portfolio_values[-1])
print("Total PnL:", np.sum(pnl))
print("Overall Sharpe Ratio:", sharpe_ratio)
print("Maximum Drawdown:", max_drawdown)

# Plotting the Portfolio Value
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label='Portfolio Value', color='blue')
plt.title('Dynamically Optimized Portfolio Value (Max Return, Volatility Constraint)')
plt.xlabel('Rebalancing Periods')
plt.ylabel('Portfolio Value')
plt.legend()
plt.show()

# Plotting Drawdown
drawdown = (np.array(portfolio_values) / np.maximum.accumulate(portfolio_values)) - 1
plt.figure(figsize=(12, 6))
plt.plot(drawdown, label='Drawdown', color='red')
plt.title('Portfolio Drawdown')
plt.xlabel('Rebalancing Periods')
plt.ylabel('Drawdown')
plt.show()

