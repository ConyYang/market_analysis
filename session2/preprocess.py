import pandas as pd

hpi = pd.read_csv("../data_session2/housing_price.csv")
rate = pd.read_csv("../data_session2/interest_rate.csv")
income = pd.read_csv("../data_session2/income.csv")
unemp = pd.read_csv("../data_session2/unemployment.csv")

hpi.columns = ['DATE', 'HousingPrice']
rate.columns = ['DATE', 'InterestRate']
income.columns = ['DATE', 'Income']
unemp.columns = ['DATE', 'Unemployment']

for df in [hpi, rate, income, unemp]:
    df['DATE'] = pd.to_datetime(df['DATE'])

merged = hpi.merge(rate, on='DATE', how='inner') \
            .merge(income, on='DATE', how='inner') \
            .merge(unemp, on='DATE', how='inner')

merged = merged.sort_values(by='DATE')

# Save to CSV
merged.to_csv("merged_housing_macro.csv", index=False)