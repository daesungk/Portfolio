import numpy as np
import matplotlib.pyplot as plt

def jump_diffusion(S0, mu, sigma, lambda_jump, mu_jump, sigma_jump, T, pathnum, stepnum):
    dt = float(T) / float(stepnum)
    paths = np.zeros((pathnum, stepnum))
    paths[:, 0] = S0  # Initial value set here
    np.random.seed(42)
    
    for t in range(1, stepnum):
        # GBM part
        z = np.random.normal(0, 1, pathnum)
        dS_gbm = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z 
        
        # Jump part
        jump_occurs = np.random.poisson(lambda_jump * dt, pathnum)
        jump_magnitude = np.where(jump_occurs > 0, 
                                  np.random.normal(mu_jump, sigma_jump, pathnum), 
                                  0)
        
        # Update paths
        paths[:, t] = paths[:, t - 1] * np.exp(dS_gbm + jump_magnitude)
    
    return paths

# Parameters    
S0 = 100     
mu = 0.05       
sigma = 0.2     
lambda_jump = 0.5  
mu_jump = 0.02     
sigma_jump = 0.1   
T = 1.0
pathnum = 10000      
stepnum = 252 

# Running the simulation
paths = jump_diffusion(S0, mu, sigma, lambda_jump, mu_jump, sigma_jump, T, pathnum, stepnum)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(paths[:100].T, lw=1)
plt.title('Monte Carlo Simulation of Jump Diffusion Model')
plt.xlabel("Time Steps")
plt.ylabel("Simulated Values")
plt.show()
