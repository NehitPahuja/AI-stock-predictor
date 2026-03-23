"""
Phase 2.1 — Feature Engineering Module
Calculates technical indicators used as input features for ML models:
  - Moving Averages (MA10, MA50)
  - Daily Returns
  - Rolling Volatility (20-day)
  - Lag features for the closing price
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add 10-day and 50-day simple moving averages of the Close price."""
    df = df.copy()
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    return df


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily percentage returns based on the Close price."""
    df = df.copy()
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def add_rolling_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add rolling standard deviation of daily returns as a volatility measure."""
    df = df.copy()
    # Ensure Daily_Return exists
    if "Daily_Return" not in df.columns:
        df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=window).std()
    return df


def add_lag_features(df: pd.DataFrame, lags: int = 5) -> pd.DataFrame:
    """Add lagged closing prices (Close_Lag_1 … Close_Lag_N) for autoregressive modelling."""
    df = df.copy()
    for lag in range(1, lags + 1):
        df[f"Close_Lag_{lag}"] = df["Close"].shift(lag)
    return df


def add_price_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum features: price change over the last 5 and 10 days."""
    df = df.copy()
    df["Momentum_5"] = df["Close"] - df["Close"].shift(5)
    df["Momentum_10"] = df["Close"] - df["Close"].shift(10)
    return df


def add_volume_change(df: pd.DataFrame) -> pd.DataFrame:
    """Add percentage change in trading volume."""
    df = df.copy()
    if "Volume" in df.columns:
        df["Volume_Change"] = df["Volume"].pct_change()
    return df


def engineer_features(df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
    """
    Master function — applies *all* feature engineering steps to the raw
    cleaned DataFrame and optionally drops rows with NaN values that result
    from rolling / lag calculations.

    Args:
        df:      Cleaned stock DataFrame (must contain 'Close' and optionally 'Volume').
        drop_na: Whether to drop rows with NaN produced by rolling windows.

    Returns:
        DataFrame enriched with all engineered features.
    """
    df = add_moving_averages(df)
    df = add_daily_returns(df)
    df = add_rolling_volatility(df)
    df = add_lag_features(df)
    df = add_price_momentum(df)
    df = add_volume_change(df)

    if drop_na:
        df.dropna(inplace=True)

    print(f"Feature engineering complete. Shape: {df.shape}")
    print(f"Feature columns: {list(df.columns)}")
    return df


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_loader import fetch_stock_data
    from src.utils import clean_stock_data

    print("--- Testing Feature Engineering ---")
    raw = fetch_stock_data("AAPL", period="6mo")
    cleaned = clean_stock_data(raw)
    featured = engineer_features(cleaned)
    print(featured.tail())
