"""
feature_engineering.py
=======================
Transforms raw credit card data into features suitable for Feast.
Computes engineered features and writes a parquet file that Feast
uses as its offline data source.

Features computed:
- v1-v28: Original PCA features (renamed to lowercase for Feast)
- scaled_amount: StandardScaler on Amount
- scaled_time: StandardScaler on Time
- hour_of_day: Hour extracted from Time (assuming start = midnight)
- is_night: 1 if transaction between 11pm-5am
- amount_zscore: Z-score of Amount

Usage:
    python src/feature_engineering.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import yaml

params = yaml.safe_load(open("params.yaml"))


def engineer_features():
    """Transform raw CSV into feature parquet for Feast."""
    raw_path = params["data"]["raw_path"]
    output_path = params["data"]["processed_path"]

    print(f"[features] Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)
    print(f"[features] Raw shape: {df.shape}")

    # ── Rename V1-V28 to lowercase (Feast convention) ─────────────────────────
    v_cols_map = {f"V{i}": f"v{i}" for i in range(1, 29)}
    df = df.rename(columns=v_cols_map)

    # ── Scale Amount ──────────────────────────────────────────────────────────
    scaler_amount = StandardScaler()
    df["scaled_amount"] = scaler_amount.fit_transform(df[["Amount"]]).astype(np.float32)

    # ── Scale Time ────────────────────────────────────────────────────────────
    scaler_time = StandardScaler()
    df["scaled_time"] = scaler_time.fit_transform(df[["Time"]]).astype(np.float32)

    # ── Hour of day (Time is seconds from first transaction) ──────────────────
    df["hour_of_day"] = ((df["Time"] / 3600) % 24).astype(int)

    # ── Is night transaction (11pm - 5am) ─────────────────────────────────────
    df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)

    # ── Amount Z-score ────────────────────────────────────────────────────────
    df["amount_zscore"] = ((df["Amount"] - df["Amount"].mean()) / df["Amount"].std()).astype(np.float32)

    # ── Add entity column and timestamp for Feast ─────────────────────────────
    df["transaction_id"] = range(1, len(df) + 1)

    # Create realistic timestamps (spread over 2 days ending near current time)
    # Use recent dates so Feast materialize-incremental picks them up
    from datetime import datetime as dt_now
    base_time = dt_now.utcnow() - timedelta(days=1)
    df["event_timestamp"] = df["Time"].apply(
        lambda t: base_time + timedelta(seconds=float(t) * (86400 / 172800))
    )

    # ── Rename target column ──────────────────────────────────────────────────
    df = df.rename(columns={"Class": "is_fraud"})

    # ── Select final columns ──────────────────────────────────────────────────
    feature_cols = [f"v{i}" for i in range(1, 29)] + [
        "scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"
    ]
    meta_cols = ["transaction_id", "event_timestamp", "is_fraud"]

    df_final = df[meta_cols + feature_cols].copy()

    # Ensure float32 for all feature columns
    for col in feature_cols:
        if col not in ["hour_of_day", "is_night"]:
            df_final[col] = df_final[col].astype(np.float32)

    # ── Save to parquet ───────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_parquet(output_path, index=False)
    print(f"[features] Feature parquet saved to {output_path}")
    print(f"[features] Shape: {df_final.shape}")
    print(f"[features] Columns: {list(df_final.columns)}")

    # ── Also save to Feast data directory ─────────────────────────────────────
    feast_data_path = "feature_repo/data/features.parquet"
    os.makedirs(os.path.dirname(feast_data_path), exist_ok=True)
    df_final.to_parquet(feast_data_path, index=False)
    print(f"[features] Feast source parquet saved to {feast_data_path}")

    # ── Print stats ───────────────────────────────────────────────────────────
    print(f"\n[features] Dataset statistics:")
    print(f"  Total transactions: {len(df_final):,}")
    print(f"  Fraud transactions: {df_final['is_fraud'].sum():,}")
    print(f"  Fraud ratio: {df_final['is_fraud'].mean():.4%}")
    print(f"  Night transactions: {df_final['is_night'].sum():,}")

    return df_final


if __name__ == "__main__":
    engineer_features()
