import streamlit as st

with st.sidebar:
    st.header("Global Controls")

    ticker = st.text_input("Ticker", "AAPL")
    forecast_horizon = st.slider("Forecast horizon (days)", 7, 90, 30)
    n_paths = st.slider("Forecast scenarios (demo)", 1, 20, 5)

    st.markdown("---")
    data_source = st.radio(
        "Data source",
        ["Fake demo data", "Upload CSV", "Yahoo Finance"],
    )

    st.markdown(
        """
        **CSV requirements (minimum):**
        - Columns: `Date`, `Close`  
        - Optional: `Open`, `High`, `Low`, `Volume`
        """
    )