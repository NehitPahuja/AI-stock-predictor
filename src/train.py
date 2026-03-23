"""
Phase 2.2 & 2.3 — Model Training Module
Trains two MVP models for next-day price prediction:
  - Baseline: Linear Regression (Phase 2.2)
  - Advanced:  Random Forest   (Phase 2.3)

Models are persisted to the ``models/`` directory using joblib so
they can be loaded later by the prediction engine.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ── Helpers ──────────────────────────────────────────────────────────────────

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Features used by each model tier
BASELINE_FEATURES = ["Close"]
ADVANCED_FEATURES = [
    "Close", "MA10", "MA50",
    "Daily_Return", "Volatility",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_4", "Close_Lag_5",
    "Momentum_5", "Momentum_10",
    "Volume", "Volume_Change",
]


def _prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the prediction target: next-day closing price."""
    df = df.copy()
    df["Target"] = df["Close"].shift(-1)   # predict NEXT day's close
    df.dropna(inplace=True)
    return df


def _split(df: pd.DataFrame, features: list[str], test_size: float = 0.2):
    """Split data into train / test sets (time-ordered, no shuffle)."""
    available = [f for f in features if f in df.columns]
    X = df[available].values
    y = df["Target"].values

    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test, available


def _evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute and print evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # Direction accuracy: did we get the move direction right?
    direction_actual = np.diff(y_true) > 0
    direction_pred = np.diff(y_pred) > 0
    min_len = min(len(direction_actual), len(direction_pred))
    dir_acc = np.mean(direction_actual[:min_len] == direction_pred[:min_len]) * 100

    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "Direction_Accuracy": round(dir_acc, 2)}
    print(f"\n📊 {name} Evaluation:")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return metrics


def _save_model(model, filename: str):
    """Persist a trained model to disk."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")


# ── Public API ───────────────────────────────────────────────────────────────

def train_linear_regression(df: pd.DataFrame, save: bool = True) -> tuple:
    """
    Phase 2.2 — Train a simple Linear Regression on the closing price only.

    Returns:
        (model, metrics_dict, feature_names)
    """
    df = _prepare_target(df)
    X_train, X_test, y_train, y_test, feat_names = _split(df, BASELINE_FEATURES)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _evaluate("Linear Regression", y_test, y_pred)

    if save:
        _save_model(model, "linear_regression.joblib")

    return model, metrics, feat_names


def train_random_forest(df: pd.DataFrame, save: bool = True, **rf_kwargs) -> tuple:
    """
    Phase 2.3 — Train a Random Forest using engineered features.

    Returns:
        (model, metrics_dict, feature_names)
    """
    df = _prepare_target(df)
    X_train, X_test, y_train, y_test, feat_names = _split(df, ADVANCED_FEATURES)

    defaults = dict(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    defaults.update(rf_kwargs)

    model = RandomForestRegressor(**defaults)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = _evaluate("Random Forest", y_test, y_pred)

    if save:
        _save_model(model, "random_forest.joblib")

    # Feature importance (nice to have for the UI later)
    importances = dict(zip(feat_names, model.feature_importances_))
    sorted_imp = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("\n🔑 Feature Importances (Random Forest):")
    for feat, imp in sorted_imp.items():
        print(f"   {feat}: {imp:.4f}")

    return model, metrics, feat_names


# ── CLI entry-point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from src.data_loader import fetch_stock_data
    from src.utils import clean_stock_data
    from src.features import engineer_features

    ticker = "AAPL"
    print(f"\n{'='*60}")
    print(f"  Training models for {ticker}")
    print(f"{'='*60}")

    raw = fetch_stock_data(ticker, period="2y")
    cleaned = clean_stock_data(raw)

    # --- Baseline (Linear Regression on raw Close) ---
    print("\n── Phase 2.2: Linear Regression ──")
    lr_model, lr_metrics, _ = train_linear_regression(cleaned)

    # --- Advanced (Random Forest on engineered features) ---
    print("\n── Phase 2.3: Random Forest ──")
    featured = engineer_features(cleaned)
    rf_model, rf_metrics, _ = train_random_forest(featured)
