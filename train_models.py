"""
train_models.py
Trains 4 forecasting models on the same data using walk-forward (time-aware)
validation, and compares them on RMSE / MAE / MAPE.

Walk-forward validation: instead of one random train/test split (which leaks
future information into training for time series), we use the last N days as
a rolling test window and only ever train on data BEFORE the test window.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

TEST_DAYS = 60  # holdout window for evaluation


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_data():
    raw = pd.read_csv("data/sales.csv", parse_dates=["date"])
    feat = pd.read_csv("data/sales_features.csv", parse_dates=["date"])
    return raw, feat


def naive_baseline(raw, test_days):
    train, test = raw.iloc[:-test_days], raw.iloc[-test_days:]
    # naive forecast = value from 7 days ago (accounts for weekly seasonality)
    preds = train["sales"].iloc[-7:].tolist()
    preds = (preds * (test_days // 7 + 1))[:test_days]
    return test["date"].values, test["sales"].values, np.array(preds)


def run_sarima(raw, test_days):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError:
        print("statsmodels not installed — skipping SARIMA")
        return None
    train, test = raw.iloc[:-test_days], raw.iloc[-test_days:]
    model = SARIMAX(train["sales"], order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    preds = fit.forecast(steps=test_days)
    return test["date"].values, test["sales"].values, preds.values


def run_prophet(raw, test_days):
    try:
        from prophet import Prophet
    except ImportError:
        print("prophet not installed — skipping Prophet")
        return None
    train, test = raw.iloc[:-test_days], raw.iloc[-test_days:]
    dfp = train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    m = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    if "holiday" in train.columns:
        m.add_regressor("holiday")
        dfp["holiday"] = train["holiday"].values
    m.fit(dfp)
    future = test.rename(columns={"date": "ds"})[["ds"]].copy()
    if "holiday" in test.columns:
        future["holiday"] = test["holiday"].values
    fcst = m.predict(future)
    return test["date"].values, test["sales"].values, fcst["yhat"].values


def run_xgboost(feat, test_days):
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("xgboost not installed — skipping XGBoost")
        return None
    train, test = feat.iloc[:-test_days], feat.iloc[-test_days:]
    drop_cols = ["date", "sales"]
    X_train, y_train = train.drop(columns=drop_cols), train["sales"]
    X_test, y_test = test.drop(columns=drop_cols), test["sales"]
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return test["date"].values, y_test.values, preds


def run_lstm(raw, test_days, lookback=14):
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
    except ImportError:
        print("tensorflow not installed — skipping LSTM")
        return None

    series = raw["sales"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    split = len(X) - test_days
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(lookback, 1)),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    preds_scaled = model.predict(X_test, verbose=0)
    preds = scaler.inverse_transform(preds_scaled).flatten()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    dates = raw["date"].values[-test_days:]
    return dates, actual, preds


def main():
    os.makedirs("outputs", exist_ok=True)
    raw, feat = load_data()

    results = {}
    predictions = {}

    for name, fn, args in [
        ("Naive (7-day)", naive_baseline, (raw, TEST_DAYS)),
        ("SARIMA", run_sarima, (raw, TEST_DAYS)),
        ("Prophet", run_prophet, (raw, TEST_DAYS)),
        ("XGBoost", run_xgboost, (feat, TEST_DAYS)),
        ("LSTM", run_lstm, (raw, TEST_DAYS)),
    ]:
        print(f"Running {name}...")
        out = fn(*args)
        if out is None:
            continue
        dates, y_true, y_pred = out
        results[name] = {
            "RMSE": round(rmse(y_true, y_pred), 2),
            "MAE": round(mean_absolute_error(y_true, y_pred), 2),
            "MAPE (%)": round(mape(y_true, y_pred), 2),
        }
        predictions[name] = (dates, y_true, y_pred)

    comparison = pd.DataFrame(results).T.sort_values("MAPE (%)")
    comparison.to_csv("outputs/comparison.csv")
    print("\n=== Model comparison (lower is better) ===")
    print(comparison)

    if "Naive (7-day)" in results:
        baseline_mape = results["Naive (7-day)"]["MAPE (%)"]
        best_model = comparison.index[0]
        if best_model != "Naive (7-day)":
            best_mape = results[best_model]["MAPE (%)"]
            improvement = round((baseline_mape - best_mape) / baseline_mape * 100, 1)
            print(f"\n{best_model} improves MAPE by {improvement}% over the naive baseline.")

    # --- plot actual vs predicted for the best model ---
    plt.figure(figsize=(12, 5))
    any_name = next(iter(predictions))
    dates, y_true, _ = predictions[any_name]
    plt.plot(dates, y_true, label="Actual", color="black", linewidth=2)
    for name, (d, _, y_pred) in predictions.items():
        plt.plot(d, y_pred, label=name, alpha=0.7)
    plt.legend()
    plt.title("Forecast comparison: Actual vs Predicted (last {} days)".format(TEST_DAYS))
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig("outputs/forecast_comparison.png", dpi=150)
    print("Saved chart to outputs/forecast_comparison.png")


if __name__ == "__main__":
    main()
