from session1.normality_test import normality_tests
from session1.process_file import standardize_data
from session1.summary_statistics import summary_statistics
from session1.uni_emp_dist import plot as uni_emp_dist_plot
from session1.scatter_plot import plot as scatter_plot

if __name__ == '__main__':
    processed_df = standardize_data("../data/TSLA_data.csv")
    df_clean = processed_df[['Return', 'LogVolume']].dropna()
    returns = df_clean['Return']
    logvol = df_clean['LogVolume']

    # Task 1: Summary statistics (mean, variance, correlation, covariance estimates etc)
    summary_statistics(processed_df)

    # Task2: Distribution test (for marginal distributions) e.g., normality test
    normality_tests(returns, "Return")
    normality_tests(logvol, "LogVolume")

    # Task 3: Plot of univariate empirical distributions
    uni_emp_dist_plot(returns, logvol)

    # Task 4: Scatter plot of the integral transformed data
    scatter_plot(returns, logvol)

