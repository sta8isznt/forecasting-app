forecasting_app/
│   app.py                      # Streamlit UI and Pipeline execution (forecast/pipeline.py)
│   requirements.txt
│
├── src/
│   ├── data/
│   │   ├── fetch.py            # Download data from sources
│   │   ├── preprocess.py       # Convert raw data to the desired format for forecasting
│   │   └── validators.py       # Optional for now
│   │
│   ├── models/
│   │   ├── baseline.py
│   │   ├── arima.py
│   │   ├── lstm.py
│   │   └── prophet.py
│   │
│   ├── forecast/
│   │   └── pipeline.py         # End to end execution
│   │
│   ├── backtest/
│   │   ├── rolling.py          # Rolling window backtesting
│   │   └── metrics.py
│   │
│   └── utils/
│       ├── plots.py
│       ├── helpers.py
│       └── config.py
│
└── docs/
    └── architecture.md
