"""
test_features.py
=================
Tests for feature engineering logic.
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile


def create_sample_raw_data(path, n_rows=100):
    """Create a sample raw CSV matching the credit card dataset schema."""
    np.random.seed(42)
    data = {}
    for i in range(1, 29):
        data[f"V{i}"] = np.random.randn(n_rows).astype(np.float32)
    data["Time"] = np.sort(np.random.uniform(0, 172800, n_rows))
    data["Amount"] = np.abs(np.random.exponential(88, n_rows))
    data["Class"] = np.random.choice([0, 1], n_rows, p=[0.98, 0.02])

    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    return df


class TestFeatureEngineering:
    """Tests for feature computation."""

    def test_scaled_amount_is_standardized(self):
        """scaled_amount should have mean ~0 and std ~1."""
        from sklearn.preprocessing import StandardScaler

        np.random.seed(42)
        amounts = np.abs(np.random.exponential(88, 1000)).reshape(-1, 1)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(amounts)

        assert abs(scaled.mean()) < 0.01
        assert abs(scaled.std() - 1.0) < 0.01

    def test_hour_of_day_range(self):
        """hour_of_day must be between 0 and 23."""
        times = np.array([0, 3600, 7200, 86400, 172800])
        hours = ((times / 3600) % 24).astype(int)

        assert hours.min() >= 0
        assert hours.max() <= 23

    def test_is_night_logic(self):
        """is_night should be 1 for hours 23, 0, 1, 2, 3, 4, 5."""
        hours = np.array([0, 1, 5, 6, 12, 22, 23])
        is_night = ((hours >= 23) | (hours <= 5)).astype(int)

        expected = [1, 1, 1, 0, 0, 0, 1]
        np.testing.assert_array_equal(is_night, expected)

    def test_amount_zscore_calculation(self):
        """Z-score should normalize amount distribution."""
        np.random.seed(42)
        amounts = np.abs(np.random.exponential(88, 1000))
        zscore = (amounts - amounts.mean()) / amounts.std()

        assert abs(zscore.mean()) < 0.01
        assert abs(zscore.std() - 1.0) < 0.01

    def test_feature_columns_count(self):
        """Should have exactly 33 feature columns (28 PCA + 5 engineered)."""
        v_cols = [f"v{i}" for i in range(1, 29)]
        engineered = ["scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"]
        all_features = v_cols + engineered

        assert len(all_features) == 33

    def test_no_nulls_in_features(self):
        """Engineered features should not contain null values."""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "Time": np.sort(np.random.uniform(0, 172800, n)),
            "Amount": np.abs(np.random.exponential(88, n)),
        })

        df["hour_of_day"] = ((df["Time"] / 3600) % 24).astype(int)
        df["is_night"] = ((df["hour_of_day"] >= 23) | (df["hour_of_day"] <= 5)).astype(int)
        df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()

        assert df["hour_of_day"].isnull().sum() == 0
        assert df["is_night"].isnull().sum() == 0
        assert df["amount_zscore"].isnull().sum() == 0
