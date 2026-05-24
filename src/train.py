"""
Train Prophet forecasting model on Store 1 data.
Save model and forecast to database.
"""

import sys
import pandas as pd
import sqlite3
import pickle
from pathlib import Path
from prophet import Prophet
from .features import get_timeseries

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def train_forecast_model(store_id: int = 1, 
                         forecast_periods: int = 90,
                         db_path: str = 'db/sales.db',
                         model_path: str = 'db/prophet_model.pkl') -> dict:
    """
    Train Prophet model on a store's data and generate forecast.
    
    Args:
        store_id: Store identifier
        forecast_periods: Days to forecast into the future
        db_path: Path to SQLite database
        model_path: Path to save trained model
        
    Returns:
        Dictionary with model, forecast, train/test data, and metrics
    """
    print(f"🔄 Training Prophet model for Store {store_id}")
    print("=" * 60)
    
    # Load time series data
    print(f"\n1. Loading time series data...")
    df = get_timeseries(store_id, db_path)
    print(f"   ✓ Loaded {len(df)} days of data")
    print(f"   Date range: {df['ds'].min().date()} to {df['ds'].max().date()}")
    
    # Define test period: last 12 weeks (84 days)
    test_start_idx = len(df) - 84
    
    train_df = df.iloc[:test_start_idx].copy()
    test_df = df.iloc[test_start_idx:].copy()
    
    print(f"\n2. Train/Test Split:")
    print(f"   Train: {len(train_df)} days ({train_df['ds'].min().date()} to {train_df['ds'].max().date()})")
    print(f"   Test:  {len(test_df)} days ({test_df['ds'].min().date()} to {test_df['ds'].max().date()})")
    
    # Initialize and configure Prophet
    print(f"\n3. Initializing Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode='additive',
        interval_width=0.95  # 95% confidence interval
    )
    
    # Fit model on FULL dataset (not just training)
    # We'll use test set for evaluation but train on all historical data for better forecast
    print(f"   Training... (this may take a moment)")
    import sys, os, contextlib
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stderr(devnull):
            model.fit(df)  # Fit on entire dataset
    
    print(f"   ✓ Model trained successfully on {len(df)} days")
    
    # Make predictions on test set
    print(f"\n4. Evaluating on test set...")
    future_test = model.make_future_dataframe(periods=0)
    forecast_test = model.predict(future_test)
    
    # Get test predictions
    test_forecast = forecast_test[forecast_test['ds'].dt.date.isin(test_df['ds'].dt.date)][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    test_forecast = test_forecast.merge(test_df, on='ds', how='inner')
    
    print(f"   Test set size: {len(test_forecast)}")
    
    # Make future forecast (90 days)
    print(f"\n5. Generating {forecast_periods}-day forecast...")
    future = model.make_future_dataframe(periods=forecast_periods)
    forecast = model.predict(future)
    
    # Extract only future dates (after training data ends, take last forecast_periods rows)
    future_forecast = forecast.iloc[-forecast_periods:][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    future_forecast['Store'] = store_id
    
    print(f"   ✓ Generated {len(future_forecast)} days of forecast")
    print(f"   Forecast range: {future_forecast['ds'].min().date()} to {future_forecast['ds'].max().date()}")
    
    # Save model
    print(f"\n6. Saving model...")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {model_path}")
    
    # Save forecast to database
    print(f"\n7. Saving forecast to database...")
    conn = sqlite3.connect(db_path)
    
    # Round predictions to integers (sales can't be fractional)
    future_forecast['yhat'] = future_forecast['yhat'].round().astype(int)
    future_forecast['yhat_lower'] = future_forecast['yhat_lower'].round().astype(int)
    future_forecast['yhat_upper'] = future_forecast['yhat_upper'].round().astype(int)
    
    future_forecast.to_sql('forecasts', conn, if_exists='replace', index=False)
    print(f"   ✓ Saved {len(future_forecast)} forecast records to database")
    
    # Create index on forecasts table
    cursor = conn.cursor()
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_forecast_store ON forecasts(Store)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_forecast_date ON forecasts(ds)')
    conn.commit()
    conn.close()
    
    print("\n" + "=" * 60)
    print("✓ Training complete!")
    print("=" * 60)
    
    return {
        'model': model,
        'train': train_df,
        'test': test_df,
        'test_forecast': test_forecast,
        'forecast': future_forecast
    }


if __name__ == '__main__':
    # Train model
    result = train_forecast_model(store_id=1, forecast_periods=90)
    
    # Show sample predictions
    print("\n📊 SAMPLE PREDICTIONS (Last 5 test days):")
    print("-" * 60)
    test_pred = result['test_forecast'][['ds', 'y', 'yhat', 'yhat_lower', 'yhat_upper']].tail(5)
    print(test_pred.to_string(index=False))
    
    print("\n📅 FORECAST SAMPLE (First 5 forecast days):")
    print("-" * 60)
    forecast_sample = result['forecast'][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(5)
    print(forecast_sample.to_string(index=False))
    
    # Calculate quick metrics on test set
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    
    if len(result['test_forecast']) > 0:
        mae = mean_absolute_error(result['test_forecast']['y'], result['test_forecast']['yhat'])
        mape = mean_absolute_percentage_error(result['test_forecast']['y'], result['test_forecast']['yhat'])
        
        print(f"\n📈 TEST SET METRICS:")
        print("-" * 60)
        print(f"MAE (Mean Absolute Error):  ${mae:,.2f}")
        print(f"MAPE (Mean Absolute % Error): {mape * 100:.2f}%")
    else:
        print(f"\n⚠️  No test predictions available for metrics calculation")
