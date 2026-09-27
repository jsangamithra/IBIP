"""
dashboard.py
Interactive Streamlit dashboard: shows historical sales, model comparison
metrics, and a forward forecast. Run with:

    streamlit run dashboard.py
"""
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")
st.title("📈 Sales Forecasting Dashboard")

DATA_PATH = "data/sales.csv"
COMPARISON_PATH = "outputs/comparison.csv"
CHART_PATH = "outputs/forecast_comparison.png"

if not os.path.exists(DATA_PATH):
    st.error("No data found. Run `python data_prep.py` first.")
    st.stop()

df = pd.read_csv(DATA_PATH, parse_dates=["date"])

col1, col2, col3 = st.columns(3)
col1.metric("Total records", f"{len(df):,}")
col2.metric("Avg daily sales", f"{df['sales'].mean():.0f}")
col3.metric("Date range", f"{df['date'].min().date()} → {df['date'].max().date()}")

st.subheader("Historical sales")
window = st.slider("Show last N days", min_value=30, max_value=len(df), value=180)
st.line_chart(df.set_index("date")["sales"].tail(window))

if os.path.exists(COMPARISON_PATH):
    st.subheader("Model comparison (walk-forward validation)")
    comparison = pd.read_csv(COMPARISON_PATH, index_col=0)
    st.dataframe(comparison, use_container_width=True)

    best_model = comparison["MAPE (%)"].idxmin()
    st.success(f"Best model by MAPE: **{best_model}** ({comparison.loc[best_model, 'MAPE (%)']}% error)")
else:
    st.info("Run `python train_models.py` to generate model comparison results.")

if os.path.exists(CHART_PATH):
    st.subheader("Actual vs Predicted (test window)")
    st.image(CHART_PATH, use_container_width=True)

st.caption("Built as an end-to-end forecasting pipeline: SARIMA · Prophet · XGBoost · LSTM, "
           "compared with walk-forward validation.")
