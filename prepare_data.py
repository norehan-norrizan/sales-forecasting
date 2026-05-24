"""
Prepare Rossmann Store Sales data for our pipeline.
Filters to stores 1-5 and saves to data/sales_data.csv.
"""

import sys
import pandas as pd
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

input_path = 'Kaggle/train.csv'
output_path = 'data/sales_data.csv'

print(f"Loading Rossmann data from {input_path}...")
df = pd.read_csv(input_path, low_memory=False)
print(f"Full dataset: {len(df):,} rows, {df['Store'].nunique()} stores")

# Filter to stores 1-5
df = df[df['Store'].isin([1, 2, 3, 4, 5])].copy()
print(f"Filtered to stores 1-5: {len(df):,} rows")

# Sort by store and date
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Store', 'Date']).reset_index(drop=True)

# Drop DayOfWeek (already derivable from Date)
df = df.drop(columns=['DayOfWeek'])

# Save
Path(output_path).parent.mkdir(exist_ok=True)
df.to_csv(output_path, index=False)

print(f"\n✓ Saved to {output_path}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"\nPreview:")
print(df.head(5))
print(f"\nMissing values:")
print(df.isnull().sum())
