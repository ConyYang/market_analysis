def summary_statistics(df):
    summary = df[['Return', 'LogVolume']].describe().T
    summary['variance'] = df[['Return', 'LogVolume']].var().values
    summary = summary[['mean', 'std', 'variance', 'min', '25%', '50%', '75%', 'max']]
    print("Summary Statistics:")
    print(summary.to_string())

    cov_matrix = df[['Return', 'LogVolume']].cov()
    corr_matrix = df[['Return', 'LogVolume']].corr()

    print("\nCovariance matrix:")
    print(cov_matrix)

    print("\nCorrelation matrix:")
    print(corr_matrix)
