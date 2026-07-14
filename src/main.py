import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def calibrate_ou_parameters(prices: np.ndarray, dt: float = 1/12):
    """
    Calibrates Ornstein-Uhlenbeck parameters using OLS regression.
    Returns: mu, theta, sigma
    """
    S_prev = prices[:-1].reshape(-1, 1)
    S_curr = prices[1:]
    
    model = LinearRegression().fit(S_prev, S_curr)
    a, b = model.intercept_, model.coef_[0]
    
    # Mathematical Recovery
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    
    residuals = S_curr - model.predict(S_prev)
    sigma_eps = np.std(residuals)
    sigma = sigma_eps * np.sqrt((2 * theta) / (1 - b**2))
    
    return mu, theta, sigma

def simulate_paths(S0, mu, theta, sigma, dt, n_sims, n_steps):
    """
    Runs Euler-Maruyama Monte Carlo simulation.
    """
    paths = np.zeros((n_steps, n_sims))
    paths[0] = S0
    
    for t in range(1, n_steps):
        Z = np.random.standard_normal(n_sims)
        paths[t] = paths[t-1] + theta * (mu - paths[t-1]) * dt + sigma * np.sqrt(dt) * Z
        
    return paths

# Add your main block here to tie it together!
if __name__ == "__main__":
    # Load data here
    # Call functions here
    print("Model calibrated successfully.")
