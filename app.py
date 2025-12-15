import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta

from src.data.fetch import fetch_yfinance
from src.data.preprocess import preprocess

# ------------------
# Page config
# ------------------
st.set_page_config(
    page_title="Stock Dashboard",
    layout="wide"
)

st.title("📈 Stock Market Dashboard")
st.markdown(
    """
    <style>
        body, .stApp {background-color: #0b0c10;}
        /* add breathing room so the title isn't clipped */
        .block-container {padding-top: 2.8rem;}
        h1 {margin-top: 0;}
        h1, h2, h3, h4, h5, h6 {color: #e8e9ea;}
        .stMarkdown, .stMetric, .stDataFrame {color: #cfd2d6;}
        .css-1offfwp, .st-emotion-cache-1dj3tuo {background: #0f1115;}
        .stSlider, .stSelectbox, .stMultiSelect, .stTextInput {color: #cfd2d6;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------
# Load & preprocess (cached)
# ------------------
@st.cache_data(ttl=60 * 60)
def download_data(tickers: tuple):
    """Download a long history once per ticker set and reuse across date filters."""
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

# Defaults: a small universe of popular tickers
default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
default_start = date.today() - timedelta(days=365 * 10)
default_end = date.today()

# ------------------
# Sidebar controls
# ------------------
st.sidebar.header("Controls")

user_tickers = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=", ".join(default_tickers),
    help="Add tickers separated by commas (e.g., AAPL, NVDA, JPM).",
)

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start,
    help="Choose a long historical start to explore cycles.",
)
end_date = st.sidebar.date_input(
    "End date",
    value=default_end,
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

# ------------------
# KPIs
# ------------------
st.subheader("Snapshot")
for chunk_start in range(0, len(selected_tickers), 3):
    cols = st.columns(3)
    for idx, ticker in enumerate(selected_tickers[chunk_start:chunk_start + 3]):
        df_t = data[data["ticker"] == ticker]
        if df_t.empty:
            continue
        last_price = df_t["close"].iloc[-1]
        first_price = df_t["close"].iloc[0]
        pct_change = (last_price / first_price - 1) * 100
        cols[idx].metric(
            ticker,
            f"{last_price:.2f}",
            f"{pct_change:+.2f}%",
            delta_color="normal",
        )

# ------------------
# Price + OHLC/Volume chart
# ------------------
st.subheader("Price / OHLC / Volume")

col_interval, col_ticker = st.columns([1, 1])
with col_interval:
    intervals = {
        "1W": 7,
        "1M": 30,
        "3M": 90,
        "6M": 182,
        "1Y": 365,
        "3Y": 365 * 3,
        "5Y": 365 * 5,
        "MAX": None,
    }
    interval_choice = st.radio(
        "Interval",
        list(intervals.keys()),
        index=5,  # default to 3Y
        horizontal=True,
    )
with col_ticker:
    detail_ticker = st.selectbox("Ticker", options=selected_tickers)

detail_df = data[data["ticker"] == detail_ticker].dropna(subset=["open", "high", "low", "close"])

if detail_df.empty:
    st.info("No OHLC data available for this ticker in the selected range.")
else:
    max_date = detail_df["date"].max()
    days = intervals[interval_choice]
    if days:
        start_cut = max_date - pd.Timedelta(days=days)
        detail_df = detail_df[detail_df["date"] >= start_cut]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[0.72, 0.28],
    )

    fig.add_trace(
        go.Candlestick(
            x=detail_df["date"],
            open=detail_df["open"],
            high=detail_df["high"],
            low=detail_df["low"],
            close=detail_df["close"],
            name=detail_ticker,
            increasing_line_color="#7fd3a8",
            decreasing_line_color="#e36464",
        ),
        row=1,
        col=1,
    )

    if "volume" in detail_df.columns and not detail_df["volume"].isna().all():
        colors = detail_df["close"].diff().apply(lambda x: "#7fd3a8" if x >= 0 else "#e36464")
        fig.add_trace(
            go.Bar(
                x=detail_df["date"],
                y=detail_df["volume"],
                marker_color=colors,
                name="Volume",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b0c10",
        plot_bgcolor="#0b0c10",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_xaxes(title="Date", row=2, col=1)
    fig.update_yaxes(title="Price", row=1, col=1)
    fig.update_yaxes(title="Volume", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)
