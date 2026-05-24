"""
Streamlit dashboard for sales forecasting.
Display actual sales, forecast, and metrics.
"""

import streamlit as st
import pandas as pd
import sqlite3
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

DB_PATH = str(Path(__file__).parent / 'db' / 'sales.db')

# Page configuration
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-title {
        font-size: 14px;
        color: #666;
        font-weight: 500;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

def load_data(store_id):
    """Load sales and forecast data from database."""
    conn = sqlite3.connect(DB_PATH)

    # Load historical sales
    sales_query = """
    SELECT Date as date, Sales as sales
    FROM sales
    WHERE Store = ? AND Open = 1
    ORDER BY Date
    """
    sales_df = pd.read_sql(sales_query, conn, params=(store_id,))
    sales_df['date'] = pd.to_datetime(sales_df['date'])

    # Load forecast
    forecast_query = """
    SELECT ds as date, yhat as forecast, yhat_lower as lower, yhat_upper as upper
    FROM forecasts
    WHERE Store = ?
    ORDER BY ds
    """
    forecast_df = pd.read_sql(forecast_query, conn, params=(store_id,))
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Load metrics
    metrics_query = "SELECT * FROM metrics WHERE Store = ?"
    metrics_df = pd.read_sql(metrics_query, conn, params=(store_id,))

    conn.close()

    return sales_df, forecast_df, metrics_df

def create_forecast_chart(sales_df, forecast_df):
    """Create interactive Plotly chart with actual + forecast + confidence bands."""
    fig = go.Figure()
    
    # Historical sales
    fig.add_trace(go.Scatter(
        x=sales_df['date'],
        y=sales_df['sales'],
        mode='lines',
        name='Historical Sales',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Sales: $%{y:,.0f}<extra></extra>'
    ))
    
    # Forecast line
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        mode='lines',
        name='Forecast',
        line=dict(color='#ff7f0e', width=2, dash='dash'),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Forecast: $%{y:,.0f}<extra></extra>'
    ))
    
    # Upper confidence interval (filled)
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['upper'],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Lower confidence interval (filled to create band)
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['lower'],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='95% Confidence Interval',
        fillcolor='rgba(255, 127, 14, 0.2)',
        hoverinfo='skip'
    ))
    
    # Layout
    fig.update_layout(
        title='Sales History & 90-Day Forecast',
        xaxis_title='Date',
        yaxis_title='Daily Sales ($)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    # Format y-axis as currency
    fig.update_yaxes(tickformat='$,.0f')
    
    return fig

def display_metric_card(title, value, context=""):
    """Display a metric card with value."""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <small>{context}</small>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================

st.title("📊 Sales Forecasting Dashboard")
st.markdown("Real-time sales forecast and analytics powered by Prophet ML model")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Store selector
    store_id = st.selectbox(
        "Select Store",
        options=[1, 2, 3, 4, 5],
        format_func=lambda x: f"Store {x}",
        help="Choose which store to view"
    )
    
    st.divider()
    st.markdown("### 📋 About")
    st.info("""
    This dashboard forecasts daily retail sales 90 days into the future using Facebook's Prophet ML model.
    
    **Data:**
    - Historical: 2 years of daily sales (2022-2024)
    - Forecast: 90 days ahead
    - Stores: 5 retail locations
    
    **Features:**
    - Seasonality patterns
    - Promotional analysis
    - Holiday impact
    """)

# Main content
try:
    # Load data
    sales_df, forecast_df, metrics_df = load_data(store_id)
    
    if len(sales_df) == 0:
        st.warning(f"⚠️ No sales data found for Store {store_id}")
        st.stop()
    
    # ============================================================
    # SECTION 1: KEY METRICS
    # ============================================================
    st.header("📈 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_sales = sales_df['sales'].mean()
        display_metric_card(
            "Average Daily Sales",
            f"${avg_sales:,.0f}",
            context=f"Over {len(sales_df)} trading days"
        )
    
    with col2:
        total_days = len(sales_df)
        display_metric_card(
            "Total Days (Historical)",
            f"{total_days:,}",
            context="Days with store open"
        )
    
    with col3:
        if len(metrics_df) > 0:
            mape = metrics_df['MAPE'].iloc[0]  # raw decimal from sklearn, e.g. 0.12 = 12%
            mape_text = f"{mape * 100:.2f}%"
            status = "✅ Good" if mape < 0.15 else "⚠️ Fair" if mape < 0.25 else "❌ Poor"
            display_metric_card(
                "Forecast Accuracy (MAPE)",
                mape_text,
                context=f"{status}"
            )
        else:
            display_metric_card(
                "Forecast Accuracy (MAPE)",
                "N/A",
                context="Not yet evaluated"
            )
    
    # ============================================================
    # SECTION 2: FORECAST CHART
    # ============================================================
    st.header("📅 Sales Forecast")
    
    if len(forecast_df) > 0:
        fig = create_forecast_chart(sales_df, forecast_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast statistics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📊 Historical Statistics")
            stats_data = {
                'Metric': ['Min Daily Sales', 'Max Daily Sales', 'Mean Daily Sales', 'Std Dev'],
                'Value': [
                    f"${sales_df['sales'].min():,.0f}",
                    f"${sales_df['sales'].max():,.0f}",
                    f"${sales_df['sales'].mean():,.0f}",
                    f"${sales_df['sales'].std():,.0f}"
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col2:
            st.subheader("🔮 Forecast Statistics")
            forecast_stats = {
                'Metric': ['Min Forecast', 'Max Forecast', 'Mean Forecast', 'CI Range'],
                'Value': [
                    f"${forecast_df['forecast'].min():,.0f}",
                    f"${forecast_df['forecast'].max():,.0f}",
                    f"${forecast_df['forecast'].mean():,.0f}",
                    f"${forecast_df['lower'].mean():,.0f} - ${forecast_df['upper'].mean():,.0f}"
                ]
            }
            st.dataframe(pd.DataFrame(forecast_stats), use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ No forecast data available. Run train.py to generate forecast.")
    
    # ============================================================
    # SECTION 3: DETAILED FORECAST TABLE
    # ============================================================
    with st.expander("📋 Detailed Forecast Table", expanded=False):
        if len(forecast_df) > 0:
            display_df = forecast_df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            display_df = display_df.rename(columns={
                'date': 'Date',
                'forecast': 'Forecast ($)',
                'lower': 'Lower 95% CI ($)',
                'upper': 'Upper 95% CI ($)'
            })
            
            # Format currency columns
            for col in ['Forecast ($)', 'Lower 95% CI ($)', 'Upper 95% CI ($)']:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    # ============================================================
    # SECTION 4: MODEL EVALUATION
    # ============================================================
    if len(metrics_df) > 0:
        with st.expander("🎯 Model Evaluation Details", expanded=False):
            metrics = metrics_df.iloc[0]
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"${metrics['MAE']:,.2f}")
                st.metric("Mean Absolute % Error (MAPE)", f"{metrics['MAPE'] * 100:.2f}%")
                st.metric("Root Mean Squared Error (RMSE)", f"${metrics['RMSE']:,.2f}")
            
            with col2:
                st.metric("Direction Accuracy", f"{metrics['Direction_Accuracy']:.1f}%")
                st.metric("Test Period Days", int(metrics['Test_Days']))
                st.metric("Evaluation Date", metrics['Eval_Date'][:10])
            
            st.info(f"""
            **Model Performance:**
            - The model predicts sales within ${metrics['MAE']:,.0f} on average
            - Accuracy of {metrics['MAPE'] * 100:.2f}% is {'EXCELLENT' if metrics['MAPE'] < 0.15 else 'GOOD' if metrics['MAPE'] < 0.25 else 'FAIR'} for daily sales forecasting
            - Tested on {int(metrics['Test_Days'])} days of unseen data
            """)
    
    # ============================================================
    # FOOTER
    # ============================================================
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 12px; margin-top: 30px;">
        <p>Sales Forecasting Dashboard | Powered by Prophet ML & Streamlit</p>
        <p>Last updated: {}</p>
    </div>
    """.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Make sure you've run: `python src/ingest.py` and `python -m src.train`")
