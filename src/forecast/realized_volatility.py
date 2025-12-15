import pandas as pd
import numpy as np
from data.preprocess import compute_log_returns

def get_realized_vol_theoritical(df: pd.DataFrame, window: int):
    """Simple function to compute the realized volatility over a specified window"""
    returns = compute_log_returns(df)
    return np.sqrt((returns ** 2).rolling(window).sum())

def get_annualized_hist_vol(df: pd.DataFrame, window: int):
    returns = compute_log_returns(df)
    return returns.rolling(window).std() * np.sqrt(252)