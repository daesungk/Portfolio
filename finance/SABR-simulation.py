import numpy as np
import matplotlib.pyplot as plt

def sabr_simulation(F0, sigma0, beta, nu, rho, T, pathnum, stepnum):
    dt = T / stepnum
    paths = np.zeros((pathnum, stepnum))
    sigma = np.zeros((pathnum, stepnum))
    paths[:, 0] = F0
    sigma[:, 0] = sigma0
    np.random.seed(42)
    
    for t in range(1, stepnum):
        # Generate correlated random variables
        z1 = np.random.normal(0, 1, pathnum)
        z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.normal(0, 1, pathnum)
        
        # Volatility Process
        sigma[:, t] = sigma[:, t - 1] * np.exp(
            -0.5 * nu ** 2 * dt + nu * np.sqrt(dt) * z2
        )
        
        # Asset Price Process (SABR)
        paths[:, t] = paths[:, t - 1] * np.exp(
            -0.5 * sigma[:, t] ** 2 * dt 
            + sigma[:, t] * (paths[:, t - 1] ** beta) * np.sqrt(dt) * z1
        )
    
    return paths, sigma

# SABR Model Parameters
F0 = 1        # Initial price
sigma0 = 0.2    # Initial volatility
beta = 0.5      # Elasticity parameter
nu = 0.3        # Volatility of volatility
rho = -0.3      # Correlation between price and volatility
T = 1.0         # Time horizon (1 year)
pathnum = 10000      
stepnum = 252 

# Running the simulation
paths, sigma = sabr_simulation(F0, sigma0, beta, nu, rho, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(paths[:100].T, lw=1, alpha=0.6)
plt.title('Monte Carlo Simulation of SABR Model')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Values")
plt.show()

