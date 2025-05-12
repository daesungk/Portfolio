import numpy as np
import matplotlib.pyplot as plt

def GBM_simulation(S_0, r, sigma, T, pathnum, stepnum):
    dt = float(T) / float(stepnum)
    np.random.seed(42)
    paths = np.zeros((pathnum, stepnum))
    paths[:, 0] = S_0  # Corrected variable name (S0 -> S_0)
    
    for t in range(1, stepnum):
        z = np.random.normal(0, 1, pathnum)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

# Parameters
S_0 = 100      
r = 0.05       
sigma = 0.2    
T = 1.0        
pathnum = 10000      
stepnum = 252     

# Running the simulation
paths = GBM_simulation(S_0, r, sigma, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(paths[:100].T, lw=1)
plt.title('Monte Carlo Simulation of GBM')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Price")
plt.show()
