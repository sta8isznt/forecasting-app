import yfinance as yf

def fetch_yfinance(tickers: list, start: str = None, end: str = None, auto_adjust: bool = False):
    """Download and return data from yfinance in a DataFrame"""

    data = yf.download(tickers, start=start, end=end, auto_adjust=auto_adjust)

    # Check if an error occured
    if data is None or data.empty:
        return

    return data