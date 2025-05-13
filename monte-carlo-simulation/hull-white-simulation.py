import numpy as np
import matplotlib.pyplot as plt

def hull_white(S0, mu, sigma0, theta, kappa, eta, T, pathnum, stepnum):
    dt = float(T) / float(stepnum)
    price_paths = np.zeros((pathnum, stepnum))
    volatility_paths = np.zeros((pathnum, stepnum))

    # Initial values
    price_paths[:, 0] = S0
    volatility_paths[:, 0] = sigma0

    np.random.seed(42)

    for t in range(1, stepnum):
        # Random variables
        z_price = np.random.normal(0, 1, pathnum)
        z_vol = np.random.normal(0, 1, pathnum)

        # Volatility update (Hull-White stochastic volatility)
        volatility_paths[:, t] = (volatility_paths[:, t - 1] +
                                  kappa * (theta - volatility_paths[:, t - 1]) * dt +
                                  eta * np.sqrt(dt) * z_vol)
        volatility_paths[:, t] = np.maximum(volatility_paths[:, t], 0)  # No negative volatility

        # Price update
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (mu - 0.5 * volatility_paths[:, t] ** 2) * dt +
            volatility_paths[:, t] * np.sqrt(dt) * z_price)

    return price_paths, volatility_paths

# Parameters
S0 = 100      # Initial price
mu = 0.05     # Expected return
sigma0 = 0.2  # Initial volatility
theta = 0.15  # Long-term mean volatility
kappa = 0.3   # Speed of mean reversion
eta = 0.05    # Volatility of volatility
T = 1.0       # Time horizon (1 year)
pathnum = 10000  # Number of simulation paths
stepnum = 252   # Number of time steps (daily)

# Running the simulation
price_paths, volatility_paths = hull_white(S0, mu, sigma0, theta, kappa, eta, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(price_paths[:100].T, lw=1)
plt.title('Monte Carlo Simulation of Hull-White Stochastic Volatility Model')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Values")
plt.show()

