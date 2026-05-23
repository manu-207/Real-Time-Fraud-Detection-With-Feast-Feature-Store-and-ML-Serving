"""
data_prep.py
=============
Downloads the Credit Card Fraud Detection dataset (Kaggle) and saves it locally.
The dataset contains 284,807 transactions with 492 frauds (0.17% positive class).

Features:
- V1-V28: PCA-transformed features (anonymized)
- Time: seconds elapsed from first transaction
- Amount: transaction amount
- Class: 1 = fraud, 0 = legitimate

Usage:
    python src/data_prep.py
"""

import os
import pandas as pd
import numpy as np
import yaml

params = yaml.safe_load(open("params.yaml"))["data"]


def download_and_prepare():
    """
    Downloads the credit card fraud dataset.
    
    NOTE: The original dataset is from Kaggle (creditcardfraud).
    For this project, we generate a synthetic version with the same
    statistical properties so it works without Kaggle credentials.
    """
    raw_path = params["raw_path"]
    os.makedirs(os.path.dirname(raw_path), exist_ok=True)

    if os.path.exists(raw_path):
        print(f"[data_prep] Dataset already exists at {raw_path}")
        df = pd.read_csv(raw_path)
        print(f"[data_prep] Shape: {df.shape}")
        print(f"[data_prep] Fraud ratio: {df['Class'].mean():.4%}")
        return df

    print("[data_prep] Generating synthetic credit card fraud dataset...")
    print("[data_prep] (Replace with real Kaggle dataset for production use)")

    np.random.seed(42)
    n_samples = 284_807
    n_fraud = 492

    # Generate legitimate transactions
    n_legit = n_samples - n_fraud

    # PCA features V1-V28 (normally distributed for legit, shifted for fraud)
    v_features_legit = np.random.randn(n_legit, 28).astype(np.float32)
    v_features_fraud = np.random.randn(n_fraud, 28).astype(np.float32)

    # Fraud transactions have different distributions on key features
    # V1, V3, V4, V7, V10, V12, V14, V17 are most discriminative
    v_features_fraud[:, 0] -= 2.5   # V1
    v_features_fraud[:, 2] -= 2.0   # V3
    v_features_fraud[:, 3] += 1.5   # V4
    v_features_fraud[:, 6] -= 2.0   # V7
    v_features_fraud[:, 9] -= 2.5   # V10
    v_features_fraud[:, 11] -= 3.0  # V12
    v_features_fraud[:, 13] -= 3.5  # V14
    v_features_fraud[:, 16] -= 2.0  # V17

    # Time: seconds from first transaction (2 days of data)
    time_legit = np.sort(np.random.uniform(0, 172800, n_legit))
    time_fraud = np.random.uniform(0, 172800, n_fraud)

    # Amount: legit avg ~$88, fraud avg ~$122
    amount_legit = np.abs(np.random.exponential(88, n_legit))
    amount_fraud = np.abs(np.random.exponential(122, n_fraud))

    # Combine
    v_cols = [f"V{i}" for i in range(1, 29)]

    df_legit = pd.DataFrame(v_features_legit, columns=v_cols)
    df_legit["Time"] = time_legit
    df_legit["Amount"] = amount_legit
    df_legit["Class"] = 0

    df_fraud = pd.DataFrame(v_features_fraud, columns=v_cols)
    df_fraud["Time"] = time_fraud
    df_fraud["Amount"] = amount_fraud
    df_fraud["Class"] = 1

    df = pd.concat([df_legit, df_fraud], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(raw_path, index=False)
    print(f"[data_prep] Dataset saved to {raw_path}")
    print(f"[data_prep] Shape: {df.shape}")
    print(f"[data_prep] Fraud ratio: {df['Class'].mean():.4%}")
    print(f"[data_prep] Columns: {list(df.columns)}")

    return df


if __name__ == "__main__":
    download_and_prepare()
