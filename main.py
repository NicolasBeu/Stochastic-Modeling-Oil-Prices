import pandas as pd
import kagglehub
import glob
import os
import numpy as np

# Import functions from your src/ folder
from src.model import calibrate_ou_parameters, calculate_half_life
from src.simulation import run_monte_carlo
from src.visualization import plot_simulation

def main():
    # 1. Data Acquisition
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("sc231997/crude-oil-price")
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the dataset.")
    
    df = pd.read_csv(csv_files[0])
    prices = df['price'].dropna().values

    # 2. Calibration
    mu, theta, sigma = calibrate_ou_parameters(prices)
    
    print(f"Calibration successful:")
    print(f"Long-term Mean (mu): {mu:.2f}")
    print(f"Mean Reversion Speed (theta): {theta:.4f}")
    print(f"Volatility (sigma): {sigma:.4f}")
    print(f"Expected Half-Life: {calculate_half_life(theta):.1f} months")

    # 3. Simulation & Visualization
    paths = run_monte_carlo(prices[-1], mu, theta, sigma, dt=1/12)
    
    # Optional: Add percentile calculations here if you want to show them in the plot
    # plot_simulation(paths, mu, ...)

if __name__ == "__main__":
    main()
