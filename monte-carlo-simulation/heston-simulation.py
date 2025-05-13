import numpy as np
import matplotlib.pyplot as plt

def heston_simulation(S0, r, kappa, theta, sigma_v, rho, v0, T, pathnum, stepnum):
    dt = float(T) / float(stepnum)
    price_paths = np.zeros((pathnum, stepnum))
    variance_paths = np.zeros((pathnum, stepnum))

    # Initial values
    price_paths[:, 0] = S0
    variance_paths[:, 0] = v0

    np.random.seed(42)

    for t in range(1, stepnum):
        # Correlated random variables
        z1 = np.random.normal(0, 1, pathnum)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, pathnum)

        # Variance update (Heston variance process)
        variance_paths[:, t] = variance_paths[:, t - 1] + kappa * (theta - variance_paths[:, t - 1]) * dt + sigma_v * np.sqrt(np.maximum(variance_paths[:, t - 1], 0)) * np.sqrt(dt) * z2
        variance_paths[:, t] = np.maximum(variance_paths[:, t], 0)  # No negative variance

        # Price update
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (r - 0.5 * variance_paths[:, t]) * dt + np.sqrt(variance_paths[:, t]) * np.sqrt(dt) * z1)

    return price_paths, variance_paths

# Parameters
S0 = 100       # Initial price
r = 0.05       # Risk-free rate
kappa = 2.0    # Speed of mean reversion
theta = 0.04   # Long-term mean variance
sigma_v = 0.3  # Volatility of variance
rho = -0.5     # Correlation between asset and variance
v0 = 0.04      # Initial variance
T = 1.0        # Time horizon (1 year)
pathnum = 10000 # Number of simulation paths
stepnum = 252   # Number of time steps (daily)

# Running the Heston simulation
price_paths, variance_paths = heston_simulation(S0, r, kappa, theta, sigma_v, rho, v0, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(price_paths[:100].T, lw=1)
plt.title('Monte Carlo Simulation of Heston Model')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Prices")
plt.show()
