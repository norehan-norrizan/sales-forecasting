"""
SQL query functions to extract features from the sales database.
"""

import pandas as pd
import sqlite3
from datetime import datetime


def get_timeseries(store_id: int, db_path: str = 'db/sales.db') -> pd.DataFrame:
    """
    Get time series data for a single store.
    Returns only days when store was open.
    
    Columns:
    - ds: Date (for Prophet)
    - y: Sales (for Prophet)
    
    Args:
        store_id: Store identifier (1-5)
        db_path: Path to SQLite database
        
    Returns:
        DataFrame with ds and y columns
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        Date as ds,
        Sales as y
    FROM sales
    WHERE Store = ? AND Open = 1
    ORDER BY Date
    """
    
    df = pd.read_sql(query, conn, params=(store_id,))
    df['ds'] = pd.to_datetime(df['ds'])
    conn.close()
    
    return df


def get_monthly_agg(store_id: int, db_path: str = 'db/sales.db') -> pd.DataFrame:
    """
    Get monthly aggregated sales for a single store.
    
    Args:
        store_id: Store identifier
        db_path: Path to SQLite database
        
    Returns:
        DataFrame with Month and aggregated Sales
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        strftime('%Y-%m', Date) as Month,
        SUM(Sales) as Total_Sales,
        AVG(Sales) as Avg_Sales,
        COUNT(*) as Days_Open
    FROM sales
    WHERE Store = ? AND Open = 1
    GROUP BY Month
    ORDER BY Month
    """
    
    df = pd.read_sql(query, conn, params=(store_id,))
    conn.close()
    
    return df


def get_promo_analysis(store_id: int, db_path: str = 'db/sales.db') -> dict:
    """
    Analyze the effect of promotions on sales for a single store.
    
    Args:
        store_id: Store identifier
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with promo vs non-promo statistics
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        Promo,
        COUNT(*) as num_days,
        AVG(Sales) as avg_sales,
        SUM(Sales) as total_sales
    FROM sales
    WHERE Store = ? AND Open = 1
    GROUP BY Promo
    """
    
    df = pd.read_sql(query, conn, params=(store_id,))
    conn.close()
    
    # Convert to dict for easier analysis
    result = {
        'with_promo': {},
        'without_promo': {}
    }
    
    for _, row in df.iterrows():
        key = 'with_promo' if row['Promo'] == 1 else 'without_promo'
        result[key] = {
            'num_days': row['num_days'],
            'avg_sales': row['avg_sales'],
            'total_sales': row['total_sales']
        }
    
    # Calculate lift (promo effect)
    if result['with_promo'] and result['without_promo']:
        promo_avg = result['with_promo']['avg_sales']
        non_promo_avg = result['without_promo']['avg_sales']
        lift = ((promo_avg - non_promo_avg) / non_promo_avg) * 100
        result['promo_lift_percent'] = lift
    
    return result


def get_store_summary(store_id: int, db_path: str = 'db/sales.db') -> dict:
    """
    Get a summary of sales stats for a single store.
    
    Args:
        store_id: Store identifier
        db_path: Path to SQLite database
        
    Returns:
        Dictionary with summary statistics
    """
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT 
        COUNT(*) as total_days,
        SUM(CASE WHEN Open = 1 THEN 1 ELSE 0 END) as days_open,
        SUM(CASE WHEN Open = 0 THEN 1 ELSE 0 END) as days_closed,
        AVG(CASE WHEN Open = 1 THEN Sales ELSE NULL END) as avg_sales,
        SUM(Sales) as total_sales,
        MIN(Sales) as min_sales,
        MAX(Sales) as max_sales
    FROM sales
    WHERE Store = ?
    """
    
    df = pd.read_sql(query, conn, params=(store_id,))
    conn.close()
    
    result = df.iloc[0].to_dict()
    return result


if __name__ == '__main__':
    print("Testing src/features.py\n")
    print("=" * 60)
    
    # Test with Store 1
    store_id = 1
    print(f"\n📊 Testing with Store {store_id}")
    print("=" * 60)
    
    # Test get_timeseries
    print("\n1. TIME SERIES (Prophet Format):")
    ts = get_timeseries(store_id)
    print(f"   Shape: {ts.shape}")
    print(f"   Date range: {ts['ds'].min().date()} to {ts['ds'].max().date()}")
    print(f"   Sales range: {ts['y'].min()} to {ts['y'].max()}")
    print("\n   First 5 rows:")
    print(ts.head())
    
    # Test get_monthly_agg
    print("\n2. MONTHLY AGGREGATION:")
    monthly = get_monthly_agg(store_id)
    print(f"   Shape: {monthly.shape}")
    print("\n   First 3 months:")
    print(monthly.head(3))
    
    # Test get_promo_analysis
    print("\n3. PROMOTION ANALYSIS:")
    promo_stats = get_promo_analysis(store_id)
    print(f"   Days with promo: {promo_stats['with_promo']['num_days']}")
    print(f"   Avg sales with promo: ${promo_stats['with_promo']['avg_sales']:.2f}")
    print(f"   Days without promo: {promo_stats['without_promo']['num_days']}")
    print(f"   Avg sales without promo: ${promo_stats['without_promo']['avg_sales']:.2f}")
    if 'promo_lift_percent' in promo_stats:
        print(f"   Promo lift: {promo_stats['promo_lift_percent']:.2f}%")
    
    # Test get_store_summary
    print("\n4. STORE SUMMARY:")
    summary = get_store_summary(store_id)
    print(f"   Total days: {summary['total_days']}")
    print(f"   Days open: {summary['days_open']}")
    print(f"   Days closed: {summary['days_closed']}")
    print(f"   Avg daily sales: ${summary['avg_sales']:.2f}")
    print(f"   Total sales: ${summary['total_sales']:,.0f}")
    print(f"   Min sales: ${summary['min_sales']}")
    print(f"   Max sales: ${summary['max_sales']}")
    
    print("\n" + "=" * 60)
    print("✓ All feature functions working correctly!")
