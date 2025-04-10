import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

def fit_gmm_fixed_weights(data, w1, w2):
    """
    Fits a 2-component Gaussian Mixture Model (GMM) to data points
    with fixed prior weights (mixing coefficients).

    Args:
        data (np.ndarray): 1D array of data points.
        w1 (float): Fixed weight/prior probability for the first Gaussian (0 < w1 < 1).
        w2 (float): Fixed weight/prior probability for the second Gaussian (w2 = 1 - w1).

    Returns:
        tuple: (mu1, sigma1, mu2, sigma2) if optimization succeeds,
               otherwise None. Returns the means and standard deviations
               of the two fitted Gaussian components.
    """
    if not np.isclose(w1 + w2, 1.0):
        raise ValueError("Weights w1 and w2 must sum to 1.")
    if not (0 < w1 < 1):
         raise ValueError("Weight w1 must be between 0 and 1 (exclusive).")
    if data is None or len(data) < 2:
        raise ValueError("Input data must not be None and contain at least two points.")

    data = np.asarray(data)

    # --- Objective Function: Negative Log-Likelihood ---
    def neg_log_likelihood(params, x_data, weight1, weight2):
        mu1, sigma1, mu2, sigma2 = params

        if sigma1 <= 1e-6 or sigma2 <= 1e-6:
            return np.inf

        pdf1 = norm.pdf(x_data, loc=mu1, scale=sigma1)
        pdf2 = norm.pdf(x_data, loc=mu2, scale=sigma2)

        combined_pdf = weight1 * pdf1 + weight2 * pdf2
        combined_pdf = np.maximum(combined_pdf, 1e-9)

        log_likelihood = np.sum(np.log(combined_pdf))

        return -log_likelihood

    # --- Initial Guesses ---
    # Simple strategy: sort data, split by weights, calculate stats
    sorted_data = np.sort(data)
    split_idx = int(len(data) * w1)
    if split_idx < 1: split_idx = 1 # Ensure at least one point per group
    if split_idx >= len(data): split_idx = len(data) - 1

    mu1_init = np.mean(sorted_data[:split_idx])
    sigma1_init = np.std(sorted_data[:split_idx], ddof=1) # Use sample std dev
    mu2_init = np.mean(sorted_data[split_idx:])
    sigma2_init = np.std(sorted_data[split_idx:], ddof=1)

    # Handle cases with zero std dev (e.g., repeated values)
    sigma1_init = max(sigma1_init, 1e-3)
    sigma2_init = max(sigma2_init, 1e-3)

    initial_params = [mu1_init, sigma1_init, mu2_init, sigma2_init]

    # --- Bounds for parameters ---
    # Means can be anything, sigmas must be positive
    bounds = [(-np.inf, np.inf), (1e-6, np.inf), (-np.inf, np.inf), (1e-6, np.inf)]

    # --- Perform Optimization ---
    result = minimize(
        neg_log_likelihood,
        initial_params,
        args=(data, w1, w2),
        method='L-BFGS-B',
        bounds=bounds
    )

    # --- Return Results ---
    if result.success:
        mu1_fit, sigma1_fit, mu2_fit, sigma2_fit = result.x
        # Ensure means are returned in a consistent order (e.g., smaller mean first)
        if mu1_fit > mu2_fit:
             # Swap results if mu1 > mu2 to maintain consistency
             return (mu2_fit, sigma2_fit, mu1_fit, sigma1_fit)
        else:
             return (mu1_fit, sigma1_fit, mu2_fit, sigma2_fit)
    else:
        print(f"Optimization failed: {result.message}")
        return None

# --- Example Usage (using data that *might* produce your histogram) ---
if __name__ == '__main__':
    # Generate some sample data resembling the histogram
    np.random.seed(42) # for reproducibility
    N1 = 6000
    N2 = 4000
    data1 = np.random.normal(loc=0, scale=0.3, size=N1)
    data2 = np.random.normal(loc=0.7, scale=0.5, size=N2)
    combined_data = np.concatenate([data1, data2])
    np.random.shuffle(combined_data)

    # --- Define your known fixed prior probabilities ---
    prior_w1 = 0.6  # Must match the ratio N1 / (N1 + N2) used above
    prior_w2 = 0.4  # Must match N2 / (N1 + N2)

    # --- Fit the GMM ---
    fit_results = fit_gmm_fixed_weights(combined_data, prior_w1, prior_w2)

    if fit_results:
        mu1, sigma1, mu2, sigma2 = fit_results
        print(f"Fitted Parameters:")
        print(f"  Gaussian 1 (Weight={prior_w1:.2f}): Mean={mu1:.4f}, Sigma={sigma1:.4f}")
        print(f"  Gaussian 2 (Weight={prior_w2:.2f}): Mean={mu2:.4f}, Sigma={sigma2:.4f}")

        # --- Optional: Visualize the result ---
        import matplotlib.pyplot as plt

        plt.figure()
        # Plot original histogram
        plt.hist(combined_data, bins=60, density=True, alpha=0.6, label='Original Histogram Data')

        # Plot fitted GMM
        x_plot = np.linspace(combined_data.min(), combined_data.max(), 500)
        pdf1_fit = prior_w1 * norm.pdf(x_plot, loc=mu1, scale=sigma1)
        pdf2_fit = prior_w2 * norm.pdf(x_plot, loc=mu2, scale=sigma2)
        gmm_fit_pdf = pdf1_fit + pdf2_fit

        plt.plot(x_plot, gmm_fit_pdf, 'r-', lw=2, label='Fitted GMM')
        plt.plot(x_plot, pdf1_fit, 'g--', label=f'Component 1 Fit')
        plt.plot(x_plot, pdf2_fit, 'm--', label=f'Component 2 Fit')

        plt.title("GMM Fit with Fixed Weights")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()
    else:
        print("Could not fit the GMM.")

