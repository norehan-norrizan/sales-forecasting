"""
Load sales CSV data into SQLite database.
Creates tables and indexes for fast querying.
"""

import sys
import pandas as pd
import sqlite3
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def ingest_data(csv_path: str, db_path: str) -> None:
    """
    Load CSV data into SQLite database.
    
    Args:
        csv_path: Path to the CSV file
        db_path: Path to the SQLite database file
    """
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    print(f"Loaded {len(df)} rows")
    
    # Connect to (or create) SQLite database
    print(f"\nCreating/connecting to SQLite database at {db_path}...")
    conn = sqlite3.connect(db_path)
    
    # Write DataFrame to SQLite
    # if_exists='replace' will overwrite if table exists
    df.to_sql('sales', conn, if_exists='replace', index=False)
    print("✓ Wrote 'sales' table to database")
    
    # Create indexes for fast querying
    cursor = conn.cursor()
    
    # Index on Store
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_store ON sales(Store)')
    print("✓ Created index on Store column")
    
    # Index on Date
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_date ON sales(Date)')
    print("✓ Created index on Date column")
    
    # Compound index on Store and Date (useful for filtering by both)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_store_date ON sales(Store, Date)')
    print("✓ Created compound index on Store and Date")
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Ingest complete! Database saved to {db_path}")


if __name__ == '__main__':
    # Paths
    csv_path = 'data/sales_data.csv'
    db_path = 'db/sales.db'
    
    # Create db directory if it doesn't exist
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Run ingest
    ingest_data(csv_path, db_path)
    
    # Verify by querying
    print("\n--- Verification ---")
    conn = sqlite3.connect(db_path)
    
    # Count rows
    row_count = pd.read_sql('SELECT COUNT(*) as count FROM sales', conn).iloc[0, 0]
    print(f"Total rows in database: {row_count}")
    
    # Show stores and date range
    store_info = pd.read_sql(
        'SELECT COUNT(DISTINCT Store) as num_stores, MIN(Date) as start_date, MAX(Date) as end_date FROM sales',
        conn
    )
    print(f"Number of stores: {store_info['num_stores'].iloc[0]}")
    print(f"Date range: {store_info['start_date'].iloc[0]} to {store_info['end_date'].iloc[0]}")
    
    # Sample data
    print("\nSample data from database:")
    sample = pd.read_sql('SELECT * FROM sales LIMIT 5', conn)
    print(sample)
    
    conn.close()
