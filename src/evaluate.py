"""
Evaluate Prophet model performance on test set.
Calculate and save metrics to database.
"""

import sys
import pandas as pd
import sqlite3
import pickle
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from .features import get_timeseries

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def evaluate_model(store_id: int = 1,
                   db_path: str = 'db/sales.db',
                   model_path: str = 'db/prophet_model.pkl') -> dict:
    """
    Evaluate trained Prophet model on test set.
    Save metrics to database.
    
    Args:
        store_id: Store identifier
        db_path: Path to SQLite database
        model_path: Path to trained model pickle
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"📊 Evaluating Prophet model for Store {store_id}")
    print("=" * 60)
    
    # Load model
    print(f"\n1. Loading trained model...")
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"   ✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"   ✗ Model not found at {model_path}")
        print(f"   Run train.py first to create the model")
        return None
    
    # Load full time series
    print(f"\n2. Loading time series data...")
    df = get_timeseries(store_id, db_path)
    print(f"   ✓ Loaded {len(df)} days")
    
    # Define test period: last 12 weeks (84 days)
    test_start_idx = len(df) - 84
    test_df = df.iloc[test_start_idx:].copy()
    print(f"   Test period: {test_df['ds'].min().date()} to {test_df['ds'].max().date()}")
    
    # Make predictions
    print(f"\n3. Generating predictions on test set...")
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    
    # Get test predictions
    test_forecast = forecast[forecast['ds'].isin(test_df['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
    test_forecast = test_forecast.merge(test_df, on='ds')
    
    print(f"   ✓ Generated predictions for {len(test_forecast)} test days")
    
    # Calculate metrics
    print(f"\n4. Calculating metrics...")
    if len(test_forecast) > 0:
        mae = mean_absolute_error(test_forecast['y'], test_forecast['yhat'])
        mape = mean_absolute_percentage_error(test_forecast['y'], test_forecast['yhat'])
        rmse = ((test_forecast['y'] - test_forecast['yhat']) ** 2).mean() ** 0.5
        
        # Mean directional accuracy: does forecast get direction right?
        actuals_change = test_forecast['y'].diff()
        forecast_change = test_forecast['yhat'].diff()
        direction_correct = ((actuals_change * forecast_change) > 0).sum()
        direction_accuracy = (direction_correct / (len(test_forecast) - 1)) * 100 if len(test_forecast) > 1 else 0
        
        metrics = {
            'Store': store_id,
            'Eval_Date': datetime.now().isoformat(),
            'Test_Days': len(test_forecast),
            'MAE': mae,
            'MAPE': mape,
            'RMSE': rmse,
            'Direction_Accuracy': direction_accuracy,
            'Min_Actual': test_forecast['y'].min(),
            'Max_Actual': test_forecast['y'].max(),
            'Mean_Actual': test_forecast['y'].mean()
        }
        
        print(f"   ✓ Metrics calculated")
    else:
        print(f"   ✗ No test forecasts available")
        return None
    
    # Save to database
    print(f"\n5. Saving metrics to database...")
    conn = sqlite3.connect(db_path)
    
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql('metrics', conn, if_exists='replace', index=False)
    print(f"   ✓ Saved metrics table")
    
    conn.close()
    
    print("\n" + "=" * 60)
    print("✓ Evaluation complete!")
    print("=" * 60)
    
    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Print formatted evaluation report."""
    if metrics is None:
        return
    
    print("\n📈 DETAILED EVALUATION REPORT")
    print("=" * 60)
    print(f"Store ID:              {metrics['Store']}")
    print(f"Evaluation Date:       {metrics['Eval_Date']}")
    print(f"Test Period Days:      {metrics['Test_Days']}")
    print("\nPrediction Accuracy:")
    print(f"  MAE (Mean Absolute Error):         ${metrics['MAE']:,.2f}")
    print(f"  MAPE (Mean Absolute % Error):      {metrics['MAPE'] * 100:.2f}%")
    print(f"  RMSE (Root Mean Squared Error):    ${metrics['RMSE']:,.2f}")
    print(f"  Direction Accuracy:                {metrics['Direction_Accuracy']:.1f}%")
    print("\nActual Sales Statistics (Test Period):")
    print(f"  Min Daily Sales:       ${metrics['Min_Actual']:,.0f}")
    print(f"  Max Daily Sales:       ${metrics['Max_Actual']:,.0f}")
    print(f"  Mean Daily Sales:      ${metrics['Mean_Actual']:,.2f}")
    print("=" * 60)


if __name__ == '__main__':
    # Evaluate model
    metrics = evaluate_model(store_id=1)
    
    if metrics:
        # Print detailed report
        print_evaluation_report(metrics)
        
        # Show interpretation
        print("\n💡 INTERPRETATION:")
        print("-" * 60)
        print(f"MAPE of {metrics['MAPE'] * 100:.2f}% means the model's predictions are")
        print(f"off by {metrics['MAPE'] * 100:.2f}% on average from actual sales.")
        print(f"\nThis is a {'GOOD' if metrics['MAPE'] < 0.15 else 'FAIR' if metrics['MAPE'] < 0.25 else 'POOR'} level of accuracy for sales forecasting.")
        print(f"\nThe model correctly predicts sales direction")
        print(f"{metrics['Direction_Accuracy']:.1f}% of the time.")
        print("=" * 60)
