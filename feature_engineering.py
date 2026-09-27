"""
feature_engineering.py
Builds lag features, rolling statistics, and calendar features used by the
ML models (XGBoost). Classical models (SARIMA/Prophet) use the raw series
directly and don't need these.
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # --- Calendar features ---
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    # --- Lag features (yesterday, last week, last month) ---
    for lag in [1, 7, 14, 30]:
        df[f"lag_{lag}"] = df["sales"].shift(lag)

    # --- Rolling statistics (based only on past data — no leakage) ---
    for window in [7, 14, 30]:
        df[f"rolling_mean_{window}"] = df["sales"].shift(1).rolling(window).mean()
        df[f"rolling_std_{window}"] = df["sales"].shift(1).rolling(window).std()

    # --- Trend feature ---
    df["days_since_start"] = (df["date"] - df["date"].min()).dt.days

    df = df.dropna().reset_index(drop=True)
    return df


if __name__ == "__main__":
    raw = pd.read_csv("data/sales.csv")
    features = build_features(raw)
    features.to_csv("data/sales_features.csv", index=False)
    print(f"Saved {len(features)} rows with {features.shape[1]} columns to data/sales_features.csv")
    print(features.columns.tolist())
