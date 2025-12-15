import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from datetime import date, timedelta

from src.data.fetch import fetch_yfinance
from src.data.preprocess import preprocess
from src.forecast.realized_volatility import (
    get_annualized_hist_vol,
    get_realized_vol_theoritical,
)

# ------------------
# Page config
# ------------------
st.set_page_config(
    page_title="Volatility Lab",
    layout="wide",
)

st.title("Volatility Lab")
st.caption("Explore rolling realized and annualized volatility with adjustable windows.")

# ------------------
# Cached helpers (reuse downloads between tweaks)
# ------------------
@st.cache_data(ttl=60 * 60)
def download_data(tickers: tuple):
    """Download a long history once per ticker set and reuse across filters."""
    return fetch_yfinance(
        list(tickers),
        start="1900-01-01",
        end=None,
        auto_adjust=False,
    )


@st.cache_data(ttl=60 * 60)
def preprocess_data(raw, tickers: tuple):
    """Preprocess raw quotes (fill/sort) and return tidy per-ticker frame."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    processed = preprocess(raw)

    tidy_frames = []
    if isinstance(processed.columns, pd.MultiIndex):
        for ticker in sorted(set(processed.columns.get_level_values(1))):
            df_t = processed.xs(ticker, level=1, axis=1)
            out = pd.DataFrame(
                {
                    "date": processed.index,
                    "ticker": ticker,
                    "open": df_t.get("Open"),
                    "high": df_t.get("High"),
                    "low": df_t.get("Low"),
                    "close": df_t["Close"] if "Close" in df_t.columns else df_t.get("Adj Close"),
                    "volume": df_t.get("Volume"),
                }
            )
            tidy_frames.append(out)
        tidy = pd.concat(tidy_frames, ignore_index=True)
    else:
        inferred_name = tickers[0] if tickers else "TICKER"
        tidy = pd.DataFrame(
            {
                "date": processed.index,
                "ticker": inferred_name,
                "open": processed.get("Open"),
                "high": processed.get("High"),
                "low": processed.get("Low"),
                "close": processed["Close"] if "Close" in processed.columns else processed.get("Adj Close"),
                "volume": processed.get("Volume"),
            }
        )

    tidy = tidy.dropna(subset=["close"]).sort_values("date")
    return tidy


def compute_vol_surfaces(data: pd.DataFrame, tickers: list, rv_window: int, hv_window: int, horizon: int):
    """Compute realized and annualized volatility series per ticker."""
    outputs = []
    for ticker in tickers:
        df_t = data[data["ticker"] == ticker].sort_values("date")
        price = df_t[["date", "close"]].dropna()
        if price.empty:
            continue

        price = price.set_index("date")
        realized = get_realized_vol_theoritical(price, rv_window).squeeze()
        annualized = get_annualized_hist_vol(price, hv_window).squeeze()

        horizon_scaled = annualized * np.sqrt(horizon / 252)

        vol_df = pd.DataFrame(
            {
                "date": price.index,
                "ticker": ticker,
                "realized_vol": realized,
                "annualized_vol": annualized,
                "horizon_vol": horizon_scaled,
            }
        ).dropna(subset=["realized_vol", "annualized_vol"])

        outputs.append(vol_df)

    if not outputs:
        return pd.DataFrame(columns=["date", "ticker", "realized_vol", "annualized_vol", "horizon_vol"])

    return pd.concat(outputs, ignore_index=True)


# ------------------
# Sidebar controls
# ------------------
st.sidebar.header("Controls")

default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
default_start = date.today() - timedelta(days=365 * 10)
default_end = date.today()

user_tickers = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(default_tickers),
    help="Add tickers separated by commas (e.g., AAPL, NVDA, JPM).",
)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
)
end_date = st.sidebar.date_input(
    "End date",
    value=default_end,
)

rv_window = st.sidebar.slider(
    "Realized window (days)",
    min_value=5,
    max_value=252,
    value=21,
    step=1,
    help="Rolling window for realized volatility.",
)

hv_window = st.sidebar.slider(
    "Annualized window (days)",
    min_value=20,
    max_value=252,
    value=63,
    step=1,
    help="Rolling window for historical annualized volatility.",
)

horizon_days = st.sidebar.slider(
    "Horizon for scaling (days)",
    min_value=1,
    max_value=90,
    value=21,
    step=1,
    help="Convert annualized vol to a shorter horizon using sqrt(h/252).",
)

if user_tickers.strip():
    requested = [t.strip().upper() for t in user_tickers.split(",") if t.strip()]
else:
    requested = default_tickers

raw = download_data(tuple(requested))
df = preprocess_data(raw, tuple(requested))
tickers = sorted(df["ticker"].unique()) if not df.empty else []

selected_tickers = st.sidebar.multiselect(
    "Select stocks",
    options=tickers,
    default=tickers[:3] if tickers else [],
)

if df.empty or not selected_tickers:
    st.info("Enter valid tickers and select at least one to load data.")
    st.stop()

# ------------------
# Filter data
# ------------------
mask = (
    df["ticker"].isin(selected_tickers) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
)

data = df.loc[mask].sort_values(["ticker", "date"])

if data.empty:
    st.info("No data available for the selected range.")
    st.stop()

# ------------------
# Compute volatility
# ------------------
vol_data = compute_vol_surfaces(
    data,
    selected_tickers,
    rv_window=rv_window,
    hv_window=hv_window,
    horizon=horizon_days,
)

if vol_data.empty:
    st.info("Not enough data to compute volatility with the chosen windows.")
    st.stop()

# ------------------
# Snapshot metrics
# ------------------
st.subheader("Latest snapshot")

latest = vol_data.sort_values("date").groupby("ticker").tail(1)
for chunk_start in range(0, len(latest), 3):
    cols = st.columns(3)
    for idx, (_, row) in enumerate(latest.iloc[chunk_start:chunk_start + 3].iterrows()):
        cols[idx].metric(
            label=row["ticker"],
            value=f"{row['annualized_vol']:.2%}",
            delta=f"Horizon {horizon_days}d: {row['horizon_vol']:.2%}",
            delta_color="inverse",
        )

# ------------------
# Charts
# ------------------
st.subheader("Rolling volatility")

plot_df = vol_data.melt(
    id_vars=["date", "ticker"],
    value_vars=["realized_vol", "annualized_vol"],
    var_name="metric",
    value_name="vol",
)

fig = px.line(
    plot_df,
    x="date",
    y="vol",
    color="ticker",
    line_dash="metric",
    labels={"vol": "Volatility", "date": "Date", "ticker": "Ticker", "metric": "Metric"},
)
fig.update_layout(
    height=500,
    legend_title=None,
    hovermode="x unified",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader(f"{horizon_days}-day scaled volatility (from annualized)")

horizon_plot = px.line(
    vol_data,
    x="date",
    y="horizon_vol",
    color="ticker",
    labels={"horizon_vol": "Scaled Volatility", "date": "Date", "ticker": "Ticker"},
)
horizon_plot.update_layout(
    height=400,
    hovermode="x unified",
)
st.plotly_chart(horizon_plot, use_container_width=True)
