#!/usr/bin/env python3
"""
Train RandomForest predictors for attention prefill/decode using Vidur-style CSVs.
Outputs:
  profiling_outputs/predictors/prefill_rf.joblib
  profiling_outputs/predictors/decode_rf.joblib
  profiling_outputs/predictors/metrics.json
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split

BASE_OUTPUT = Path(os.environ.get("PROFILE_OUTPUT_DIR", "profiling_outputs"))
ATTENTION_CSV = BASE_OUTPUT / "attention" / "attention.csv"
PREDICTOR_DIR = BASE_OUTPUT / "predictors"
PREDICTOR_DIR.mkdir(parents=True, exist_ok=True)


def _safe_relative_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    eps = 1e-9
    denom = np.where(np.abs(y_true) < eps, eps, y_true)
    return np.abs(y_true - y_pred) / np.abs(denom)


def _train_rf(X: np.ndarray, y: np.ndarray, model_name: str):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
        )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    rel_errors = _safe_relative_errors(y_test, y_pred)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_absolute_error")
    cv_rmse = np.sqrt(-cross_val_score(rf, X, y, cv=kf, scoring="neg_mean_squared_error"))

    metrics = {
        "mae": float(mae),
        "rmse": float(rmse),
        "mape_mean": float(np.mean(rel_errors)),
        "mape_p95": float(np.percentile(rel_errors, 95)),
        "mape_p99": float(np.percentile(rel_errors, 99)),
        "cv_mae_mean": float(np.mean(cv_mae)),
        "cv_mae_std": float(np.std(cv_mae)),
        "cv_rmse_mean": float(np.mean(cv_rmse)),
        "cv_rmse_std": float(np.std(cv_rmse)),
        "n_samples": int(len(X)),
    }

    model_path = PREDICTOR_DIR / f"{model_name}.joblib"
    joblib.dump(rf, model_path)
    metrics["model_path"] = str(model_path)
    return metrics


def build_prefill_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for _, row in df[df["type"] == "prefill"].iterrows():
        seq_len = row["seq_len"]
        batch = row["batch_size"]
        num_layers = row["num_layers"]
        hidden = row["hidden_size"]
        num_tokens = seq_len * batch
        effective_length = math.sqrt(max(num_tokens, 1))
        attention_feature = (effective_length ** 2) * num_layers
        mlp_feature = num_tokens * num_layers
        rows.append(
            [
                attention_feature,
                mlp_feature,
                batch,
                seq_len,
                hidden,
            ]
        )
        targets.append(row["time_mean_ms"])
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


def build_decode_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    targets = []
    for _, row in df[df["type"] == "decode"].iterrows():
        seq_len = row["seq_len"]
        batch = row["batch_size"]
        kv_cache = row["kv_cache"]
        num_layers = row["num_layers"]
        hidden = row["hidden_size"]
        decode_tokens = seq_len * batch
        total_kv = kv_cache * batch
        rows.append(
            [
                total_kv * num_layers,
                decode_tokens * num_layers,
                batch,
                kv_cache,
                hidden,
            ]
        )
        targets.append(row["time_mean_ms"])
    return np.array(rows, dtype=np.float64), np.array(targets, dtype=np.float64)


def main() -> None:
    if not ATTENTION_CSV.exists():
        raise FileNotFoundError(f"Missing attention CSV at {ATTENTION_CSV}")

    df = pd.read_csv(ATTENTION_CSV)

    metrics_summary = {}

    X_prefill, y_prefill = build_prefill_features(df)
    if len(X_prefill) < 5:
        raise RuntimeError("Not enough prefill samples to train a predictor.")
    metrics_summary["prefill"] = _train_rf(X_prefill, y_prefill, "prefill_rf")

    X_decode, y_decode = build_decode_features(df)
    if len(X_decode) < 5:
        raise RuntimeError("Not enough decode samples to train a predictor.")
    metrics_summary["decode"] = _train_rf(X_decode, y_decode, "decode_rf")

    metrics_path = PREDICTOR_DIR / "metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Saved predictor metrics to {metrics_path}")


if __name__ == "__main__":
    main()
