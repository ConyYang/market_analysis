from scipy import stats
import numpy as np

def normality_tests(x, name="Variable"):
    print(f"\nNormality tests for {name}:")

    # Shapiro–Wilk test
    stat, p = stats.shapiro(x)
    print(f"Shapiro-Wilk test: stat={stat:.4f}, p-value={p:.4f}")

    # Kolmogorov–Smirnov test (compare to N(mean, std))
    standardized = (x - np.mean(x)) / np.std(x)
    stat, p = stats.kstest(standardized, 'norm')
    print(f"K-S test: stat={stat:.4f}, p-value={p:.4f}")