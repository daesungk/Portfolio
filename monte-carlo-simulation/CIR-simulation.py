import numpy as np
import matplotlib.pyplot as plt

def cir_simulation(mu, theta, sigma, T, pathnum, stepnum):
    dt = float(T) / float(stepnum)
    paths = np.zeros((pathnum, stepnum))
    paths[:, 0] = mu
    np.random.seed(42)
    for t in range(1, stepnum):
        z = np.random.normal(0, 1, pathnum)
        paths[:,t] = paths[:,t - 1] + theta * (mu - paths[:,t - 1]) * dt + sigma * np.sqrt(np.maximum(paths[:,t - 1],0)) * np.sqrt(dt) * z
    
    return paths

# Parameters
mu = 1      
theta = 0.5   
sigma = 0.2   
T = 1.0       
pathnum = 10000      
stepnum = 252        

# Running the simulation
paths = cir_simulation(mu, theta, sigma, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(paths[:100].T, lw=1)
plt.title('Monte Carlo Simulation of CIR model')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Values")
plt.show()
