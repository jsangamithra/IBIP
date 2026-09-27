"""
data_prep.py
Generates a realistic synthetic daily sales dataset (3 years) with:
- upward trend
- weekly seasonality (weekend lift)
- yearly seasonality (holiday season lift)
- random promo events
- holiday spikes
- noise

If you have real data, skip this script and just place your CSV at
data/sales.csv with columns: date, sales, promo (0/1), holiday (0/1)
"""
import numpy as np
import pandas as pd
import os

np.random.seed(42)

def generate_synthetic_sales(start="2022-01-01", periods=3 * 365):
    dates = pd.date_range(start=start, periods=periods, freq="D")

    trend = np.linspace(200, 400, periods)                     # slow growth
    weekly = 40 * np.sin(2 * np.pi * dates.dayofweek / 7)       # weekly pattern
    yearly = 60 * np.sin(2 * np.pi * dates.dayofyear / 365.25)  # yearly pattern

    # random promo days (~8% of days), each boosts sales ~25%
    promo = np.random.binomial(1, 0.08, periods)
    promo_effect = promo * np.random.uniform(60, 120, periods)

    # holiday spikes: last week of Nov + all of Dec (Black Friday/Xmas style)
    holiday = ((dates.month == 12) | ((dates.month == 11) & (dates.day >= 24))).astype(int)
    holiday_effect = holiday * np.random.uniform(80, 160, periods)

    noise = np.random.normal(0, 25, periods)

    sales = trend + weekly + yearly + promo_effect + holiday_effect + noise
    sales = np.clip(sales, 20, None).round(2)

    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "promo": promo,
        "holiday": holiday,
    })
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_synthetic_sales()
    df.to_csv("data/sales.csv", index=False)
    print(f"Saved {len(df)} rows to data/sales.csv")
    print(df.head())
