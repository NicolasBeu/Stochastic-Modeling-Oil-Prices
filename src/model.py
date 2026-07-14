import numpy as np
from sklearn.linear_model import LinearRegression

def calibrate_ou_parameters(prices: np.ndarray, dt: float = 1/12):
    """Kalibriert die OU-Prozess-Parameter mittels OLS-Regression."""
    S_prev = prices[:-1].reshape(-1, 1)
    S_curr = prices[1:]
    
    model = LinearRegression().fit(S_prev, S_curr)
    a, b = model.intercept_, model.coef_[0]
    
    # Mathematische Parameter-Rekonstruktion
    theta = -np.log(b) / dt
    mu = a / (1 - b)
    
    residuals = S_curr - model.predict(S_prev)
    sigma_eps = np.std(residuals)
    sigma = sigma_eps * np.sqrt((2 * theta) / (1 - b**2))
    
    return mu, theta, sigma

def calculate_half_life(theta: float):
    """Berechnet die Halbwertszeit in Monaten."""
    return (np.log(2) / theta) * 12
