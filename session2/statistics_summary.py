import pandas as pd

def summary_statistics(df, print_name):
    col_name = df.columns[1]
    summary = df[col_name].describe()

    print("\nSummary statistics for", print_name)
    print(summary)
    print("\nAdditional metrics:")
    print("Variance:", df[col_name].var())
    print("Skewness:", df[col_name].skew())
    print("Kurtosis:", df[col_name].kurt())

if __name__ == '__main__':
    summary_statistics(pd.read_csv("../data_session2/income.csv"), "Income")
    summary_statistics(pd.read_csv("../data_session2/interest_rate.csv"), "Interest Rate")
    summary_statistics(pd.read_csv("../data_session2/unemployment.csv"), "Unemployment")
    summary_statistics(pd.read_csv("../data_session2/housing_price.csv"), "Housing Price")