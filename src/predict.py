"""
Phase 2.4 — Prediction Engine
Orchestrates the full pipeline:
  1. Fetch latest data for a ticker
  2. Clean & engineer features
  3. Load trained models (or train on-the-fly if none exist)
  4. Return formatted prediction output including:
     - Predicted next-day price
     - Confidence percentage
     - Bullish / Bearish conviction label
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional

from src.data_loader import fetch_stock_data
from src.utils import clean_stock_data
from src.features import engineer_features
from src.train import (
    train_linear_regression,
    train_random_forest,
    BASELINE_FEATURES,
    ADVANCED_FEATURES,
    MODELS_DIR,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_model(filename: str):
    """Load a model from disk; returns None if it doesn't exist."""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _confidence_pct(predicted: float, current: float) -> float:
    """
    Heuristic confidence score (0-100%).
    Based on the magnitude of the predicted move relative to the current price.
    Larger predicted moves → higher conviction; capped at 95%.
    """
    if current == 0:
        return 0.0
    move_pct = abs(predicted - current) / current * 100
    # Map: 0% move → 50% confidence, 5%+ move → ~95%
    confidence = min(50 + move_pct * 9, 95.0)
    return round(confidence, 1)


def _conviction_label(predicted: float, current: float) -> str:
    """Return 'Bullish' if price is predicted to rise, else 'Bearish'."""
    return "Bullish 🟢" if predicted >= current else "Bearish 🔴"


# ── Public API ───────────────────────────────────────────────────────────────

def predict(
    ticker: str,
    period: str = "2y",
    model_type: str = "random_forest",
    retrain: bool = False,
) -> dict:
    """
    End-to-end prediction for a given stock ticker.

    Args:
        ticker:     Stock symbol (e.g. 'AAPL', 'RELIANCE.NS').
        period:     Historical data period for training / context.
        model_type: 'linear_regression' or 'random_forest'.
        retrain:    Force re-training even if a saved model exists.

    Returns:
        dict with keys:
          - ticker
          - current_price
          - predicted_price
          - price_change       (absolute)
          - price_change_pct   (%)
          - confidence         (%)
          - conviction         ('Bullish 🟢' / 'Bearish 🔴')
          - model_used
          - metrics            (dict with MAE, RMSE, Direction_Accuracy)
    """
    # 1. Fetch & clean
    raw = fetch_stock_data(ticker, period=period)
    if raw.empty:
        return {"error": f"No data found for {ticker}"}
    cleaned = clean_stock_data(raw)

    # 2. Engineer features (always needed for Random Forest)
    featured = engineer_features(cleaned.copy(), drop_na=True)

    # 3. Load or train model
    model = None
    metrics = {}
    feature_names = []

    if model_type == "linear_regression":
        model_file = "linear_regression.joblib"
        if not retrain:
            model = _load_model(model_file)
        if model is None:
            model, metrics, feature_names = train_linear_regression(cleaned)
        else:
            feature_names = BASELINE_FEATURES
    else:  # random_forest (default)
        model_file = "random_forest.joblib"
        if not retrain:
            model = _load_model(model_file)
        if model is None:
            model, metrics, feature_names = train_random_forest(featured)
        else:
            feature_names = ADVANCED_FEATURES

    # 4. Build the feature vector for the *latest* row
    if model_type == "linear_regression":
        source_df = cleaned
    else:
        source_df = featured

    available_features = [f for f in feature_names if f in source_df.columns]
    latest_row = source_df[available_features].iloc[-1:].values

    # 5. Predict
    predicted_price = float(model.predict(latest_row)[0])
    current_price = float(source_df["Close"].iloc[-1])
    price_change = round(predicted_price - current_price, 2)
    price_change_pct = round((price_change / current_price) * 100, 2) if current_price else 0.0

    result = {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "predicted_price": round(predicted_price, 2),
        "price_change": price_change,
        "price_change_pct": price_change_pct,
        "confidence": _confidence_pct(predicted_price, current_price),
        "conviction": _conviction_label(predicted_price, current_price),
        "model_used": model_type,
        "metrics": metrics,
    }

    return result


def predict_multiple(
    tickers: list[str],
    model_type: str = "random_forest",
    retrain: bool = False,
) -> list[dict]:
    """Run predictions for a batch of tickers."""
    results = []
    for t in tickers:
        print(f"\n{'─'*50}")
        res = predict(t, model_type=model_type, retrain=retrain)
        results.append(res)
    return results


def format_prediction(result: dict) -> str:
    """Pretty-print a single prediction result for the terminal."""
    if "error" in result:
        return f"❌ {result['error']}"

    sign = "+" if result["price_change"] >= 0 else ""
    lines = [
        f"",
        f"═══════════════════════════════════════════",
        f"  📈  {result['ticker']}  —  Prediction Summary",
        f"═══════════════════════════════════════════",
        f"  Current Price   :  ${result['current_price']:.2f}",
        f"  Predicted Price  :  ${result['predicted_price']:.2f}",
        f"  Change           :  {sign}${result['price_change']:.2f}  ({sign}{result['price_change_pct']}%)",
        f"  Confidence       :  {result['confidence']}%",
        f"  Conviction       :  {result['conviction']}",
        f"  Model            :  {result['model_used']}",
        f"═══════════════════════════════════════════",
        f"",
    ]
    return "\n".join(lines)


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    demo_tickers = ["AAPL", "MSFT", "NVDA", "RELIANCE.NS"]

    print("\n🚀 Quantum Ledger — Prediction Engine\n")

    for ticker in demo_tickers:
        result = predict(ticker, retrain=True)
        print(format_prediction(result))
