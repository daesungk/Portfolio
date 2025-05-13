import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from sklearn.model_selection import ParameterGrid

#### Functions
## Import data
def get_ticker_prices(tickers, start, end):
    data_sample = yf.download(
    tickers,
    start=start,
    end=end,
    progress=False
    )
    prices = data_sample["Close"][tickers].dropna()
    return prices

## Find hedge ratio
def hedge_ratio_OLS(prices):
	X = sm.add_constant(np.log(prices.iloc[:,0]))
	y = np.log(prices.iloc[:,1])
	ols = sm.OLS(y,X).fit()
	return ols.params.iloc[1]

## Create the spread using the hedge ratio $\beta$
def hedged_spread(prices, beta):
    spread = np.log(prices.iloc[:, 1]) - beta * np.log(prices.iloc[:, 0])
    return spread

## ADF Test: returns stat, $p$-value, half-life, volatility
def adf_test(prices, beta):
    # Calculate spread
    spread = hedged_spread(prices,beta)
    
    # Perform ADF test
    adf = adfuller(spread, maxlag=1, regression='c')
    adf_stat = adf[0]
    pval = adf[1]
    critical_values = adf[4]
    
    # Calculate volatility
    spread_vol = spread.std()
    
    # Calculate half-life
    spread_lag = spread.shift(1).dropna()
    spread_delta = spread.diff().dropna()
    
    # Align spread_lag and spread_delta
    spread_lag = spread_lag.iloc[1:]
    spread_delta = spread_delta.iloc[1:]
    
    adf_OLS = sm.OLS(spread_delta, sm.add_constant(spread_lag)).fit()
    
    gamma = adf_OLS.params.iloc[1]
    
    if gamma < 0:  # Ensure gamma is negative for stationarity
        half_life = -np.log(2) / gamma
    else:
        half_life = np.nan
    
    # Display results
    print(f"ADF Statistic: {adf_stat}")
    print(f"p-value: {pval}")
    print(f"Critical Values: {critical_values}")
    print(f"Volatility: {spread_vol}")
    print(f"Half Life: {half_life}")

    return [adf_stat, pval, critical_values, spread_vol, half_life]

## Mean reverting strategy: returns PnL
def mean_reverting(spread, window, bband):
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std()
    zscore = (spread - mu) / sigma
    # Generate signals
    signals = pd.Series(0.0, index=spread.index)
    signals[zscore >  bband] = -1.0
    signals[zscore < - bband] = +1.0
    signals[zscore.abs() < 0.2] = 0.0
    signals = signals.ffill().fillna(0.0)
    # Compute PnL
    delta_spread = spread.diff()
    pnl = signals.shift(1) * delta_spread
    return pnl

## Print the return
def print_pnl(pnl):
    # Performance metrics
    cum_return = pnl.cumsum()
    sharpe     = pnl.mean() / pnl.std() * np.sqrt(252)
    drawdown   = cum_return - cum_return.cummax()
    max_dd     = drawdown.min()

    # Plot cumulative return
    fig, ax = plt.subplots()
    ax.plot(cum_return.index, cum_return)
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative PnL')
    ax.set_title(f'Cumulative PnL\nSharpe: {sharpe:.2f}, Max DD: {max_dd:.2f}')
    plt.tight_layout()
    plt.show()
    
    # Plot drawdown
    fig, ax = plt.subplots()
    ax.plot(drawdown.index, drawdown)
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.set_title('Drawdown Over Time')
    plt.tight_layout()
    plt.show()

## Grid Search for finding best parameters
def grid_search_params(spread, param_grid):
    # Grid search for the best Sharpe ratio
    best_sharpe = -np.inf
    best_params = None
    results = []    
    for params in ParameterGrid(param_grid):
        pnl = mean_reverting(spread, params["window"], params["bband"])
        sharpe = pnl.mean() / pnl.std() * np.sqrt(252) if pnl.std() != 0 else -np.inf
        results.append((params["window"], params["bband"], sharpe))
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params
    results_df = pd.DataFrame(results, columns=["Window", "Bband", "Sharpe"])
    # Print the best parameters and Sharpe ratio
    print("Best Parameters:", best_params)
    print("Best Sharpe Ratio:", best_sharpe)
    # Display the results
    print(results_df.sort_values(by="Sharpe", ascending=False).head(10))
    return best_params

#### Test

## Tickers
tickers = ["XOM", "CVX"]

## Historical data
#### Find hedge ratio
start = "2015-01-01"
end = "2017-12-31"
prices = get_ticker_prices(tickers, start, end)
beta = hedge_ratio_OLS(prices)
adf_result = adf_test(prices,beta)
spread = hedged_spread(prices,beta)
#### PnL
pnl = mean_reverting(spread,40,1.0)
print_pnl(pnl)
#### GridSearch
param_grid = {
    "window": range(30, 50, 4),
    "bband": np.arange(1.0, 2.5, 0.2)
}
best_params = grid_search_params(spread, param_grid)
best_pnl = mean_reverting(spread, best_params["window"], best_params["bband"])
print_pnl(best_pnl)

## Backtest
prices_bt = get_ticker_prices(tickers, "2018-01-01", "2019-12-31")
spread_bt = hedged_spread(prices_bt,beta)
pnl_bt = mean_reverting(spread_bt, best_params["window"], best_params["bband"])
print_pnl(pnl_bt)
