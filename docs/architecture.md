# Architecture of the Repo

## app.py
*Streamlit UI and Pipeline execution*
1. Input: (Forecast type), Ticker, horizon, Model
2. Calls Fetch -> Preprocess -> Forecast -> Plot