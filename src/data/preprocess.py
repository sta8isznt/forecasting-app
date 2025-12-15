import pandas as pd
import numpy as np

def fill_missing_dates(df, freq="D"):
    """Forward fill the missing dates"""
    return df.resample(freq).ffill()

def select_ticker(df, ticker):
    """Select a specific ticker from the MultiIndex DataFrame"""
    return df.xs(ticker, level=1, axis=1)

def fill_missing_values(df):
    """Fill all NaNs: trailing and leading"""
    df = df.ffill()

    # Make sure that no leading NaNs survive -> Backfill
    df = df.bfill()
    return df

def select_target_column(df, target_column):
    """Return the values from a specific column"""
    return df[[target_column]]

def compute_returns(df, column="Close"):
    """Return the returns of a series"""
    return df[[column]].pct_change().dropna()

def compute_log_returns(df):
    """Calculates the log returns of the series"""
    return np.log(df/df.shift(1))

def preprocess(df, ticker=None, target_column=None):
    df = df.sort_index()
    df = fill_missing_dates(df)
    df = fill_missing_values(df)

    if ticker is not None and target_column is not None:
        return select_target_column(select_ticker(df, ticker), target_column)
    elif ticker is not None:
        return select_ticker(df, ticker)
    elif target_column is not None:
        return select_target_column(df, target_column)

    return df