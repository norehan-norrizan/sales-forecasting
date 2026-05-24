# Sales Forecasting Dashboard

A beginner ML portfolio project — forecasts daily retail sales for 5 stores, 90 days into the future, with a Streamlit dashboard.

## Tech Stack

Python 3.10+ · pandas · Prophet · SQLite · Streamlit · Plotly · scikit-learn

## Project Structure

```
├── data/
│   └── sales_data.csv       # Generated dummy data (2 years, 5 stores)
├── db/
│   ├── sales.db             # SQLite database (generated)
│   └── prophet_model.pkl    # Trained model (generated)
├── src/
│   ├── ingest.py            # Load CSV → SQLite
│   ├── features.py          # SQL query helpers
│   ├── train.py             # Train Prophet model + save forecast
│   └── evaluate.py          # Calculate MAE/MAPE metrics
├── app.py                   # Streamlit dashboard
├── generate_data.py         # Generate dummy CSV
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## Running the Pipeline

```bash
# 1. Generate dummy data
python generate_data.py

# 2. Load data into SQLite
python -m src.ingest

# 3. Train Prophet model (Store 1, 90-day forecast)
python -m src.train

# 4. Evaluate model on test set
python -m src.evaluate

# 5. Launch dashboard
streamlit run app.py
```

## Dashboard Features

- Store selector (1–5)
- Plotly chart: historical sales + 90-day forecast + 95% confidence interval
- Metric cards: average daily sales, total trading days, MAPE score
- Expandable forecast table and model evaluation details
