import pandas as pd
import numpy as np

def standardize_data(filename):
    df = pd.read_csv(filename)

    df = df.drop([0, 1])
    df = df.reset_index(drop=True)

    df.head()

    # convert float to decimal for Close and Volume
    df[['Close', 'Volume']] = df[['Close', 'Volume']].apply(pd.to_numeric, errors='coerce')

    # daily log return
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))

    # log volume
    df['LogVolume'] = np.log(df['Volume'])
    df.dropna(subset=['Return', 'LogVolume'])
    return df