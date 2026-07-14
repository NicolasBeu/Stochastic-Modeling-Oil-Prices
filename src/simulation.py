import numpy as np

def run_monte_carlo(S0, mu, theta, sigma, dt, n_sims=5000, n_steps=120):
    """Führt die Euler-Maruyama-Simulation durch."""
    paths = np.zeros((n_steps, n_sims))
    paths[0] = S0
    
    for t in range(1, n_steps):
        Z = np.random.standard_normal(n_sims)
        paths[t] = paths[t-1] + theta * (mu - paths[t-1]) * dt + sigma * np.sqrt(dt) * Z
        
    return paths
