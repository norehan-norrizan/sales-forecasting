"""
Generate realistic dummy sales data for 5 stores over 2 years.
Features:
- Weekly seasonality (weekends have higher sales)
- Yearly seasonality (seasonal patterns)
- Promotional effects
- Random store closures (~5%)
- Correlated customers with sales
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Date range: 2 years of daily data
start_date = datetime(2022, 1, 1)
end_date = datetime(2023, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Create data for 5 stores
stores = [1, 2, 3, 4, 5]
data = []

for store in stores:
    for date in date_range:
        # Base sales with some randomness
        base_sales = 5500
        
        # Weekly seasonality: weekends (Sat=5, Sun=6) have ~20% higher sales
        day_of_week = date.dayofweek
        weekend_boost = 1.2 if day_of_week in [5, 6] else 1.0
        
        # Yearly seasonality: higher in Dec/Jan, lower in summer
        month = date.month
        if month in [12, 1]:
            seasonal_factor = 1.15
        elif month in [6, 7, 8]:
            seasonal_factor = 0.9
        else:
            seasonal_factor = 1.0
        
        # Random noise
        noise = np.random.normal(1.0, 0.1)  # ~10% variance
        
        # Promotion effect: ~15% of days have promotion, +25% sales boost
        promo = 1 if np.random.random() < 0.15 else 0
        promo_boost = 1.25 if promo else 1.0
        
        # Store is open ~95% of the time (5% closure rate)
        is_open = 1 if np.random.random() < 0.95 else 0
        
        # Calculate sales (only if open)
        if is_open:
            sales = int(base_sales * weekend_boost * seasonal_factor * promo_boost * noise)
            # Clamp to reasonable range
            sales = max(3000, min(9000, sales))
        else:
            sales = 0
        
        # Customers correlated with sales: roughly sales / 40 + some noise
        if is_open:
            customers = int(sales / 40 + np.random.normal(0, 20))
            customers = max(50, customers)
        else:
            customers = 0
        
        # State holidays: mostly "0", occasionally "a" (e.g., Easter, Christmas)
        # Simplified: "a" on Dec 25 and a few other dates
        state_holiday = "0"
        if (date.month == 12 and date.day == 25) or \
           (date.month == 1 and date.day == 1) or \
           (date.month == 4 and 10 <= date.day <= 15):  # Easter period
            state_holiday = "a"
        
        # School holidays: random, ~15% of days
        school_holiday = 1 if np.random.random() < 0.15 else 0
        
        data.append({
            'Store': store,
            'Date': date,
            'Sales': sales,
            'Customers': customers,
            'Open': is_open,
            'Promo': promo,
            'StateHoliday': state_holiday,
            'SchoolHoliday': school_holiday
        })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_path = 'data/sales_data.csv'
df.to_csv(output_path, index=False)

print(f"✓ Generated {len(df)} rows of sales data for {len(stores)} stores")
print(f"✓ Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"✓ Saved to: {output_path}")
print("\nData preview:")
print(df.head(10))
print("\nData summary:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
