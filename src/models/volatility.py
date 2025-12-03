from src.data.preprocess import compute_returns

def rv_forecast(returns, window=30):
    """
    Naive model to forecast the upcoming volatility.
    It will be used as a baseline.
    """

def ewma_forecast(returns, l):
    """Use EWMA model to forecast volatility"""

def garch_forecast(returns, q, p):
    """Use GARCH model to forecast volatility"""

def forecast_volatility(returns, model="garch", horizon=1):
    """Gives the choice for all the models"""
    if model == "historical":
        return rv_forecast(returns)
    elif model == "ewma":
        return ewma_forecast(returns, 0.94)
    else:
        return garch_forecast(returns, 1, 1)