import matplotlib.pyplot as plt

def plot_historical_data(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['price'], label='Historical Crude Oil Price', color='#004C99', linewidth=1)
    plt.axhline(y=57.27, color='r', linestyle='--', label=r'Long-term Mean ($\mu = 57.27$)')
    plt.title('Historical Crude Oil Prices (1983 - Present)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD per Barrel)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.savefig('historical_plot.png') # Save it to include in your README!
    plt.show()

def plot_calibration(S_t_minus_1, S_t, model, mu, b):
    plt.figure(figsize=(10, 6))
    plt.scatter(S_t_minus_1, S_t, alpha=0.4, s=10, label='Historical Data')
    x_vals = np.array([np.min(S_t_minus_1), np.max(S_t_minus_1)]).reshape(-1, 1)
    plt.plot(x_vals, model.predict(x_vals), color='red', linewidth=3, label=f'OLS Fit (b={b:.4f})')
    plt.plot(x_vals, x_vals, color='black', linestyle='--', label='Identity Line')
    plt.scatter(mu, mu, color='green', s=100, zorder=5, label=f'Mean ($\mu={mu:.2f}$)')
    plt.title('OU Process Calibration: Mean Reversion Evidence')
    plt.legend()
    plt.show()

def plot_simulation(simulated_paths, mean_path, p5, p95, mu, last_price):
    plt.figure(figsize=(12, 6))
    plt.plot(simulated_paths[:, :100], color='gray', alpha=0.1)
    plt.plot(mean_path, color='black', linewidth=2, label='Expected Path')
    plt.plot(p5, color='green', linestyle=':', label='5th Percentile')
    plt.plot(p95, color='blue', linestyle=':', label='95th Percentile')
    plt.axhline(y=mu, color='red', linestyle='--', label='Mean')
    plt.title(f'OU Oil Price Simulation (Initial Price = ${last_price:.2f})')
    plt.legend()
    plt.show()
