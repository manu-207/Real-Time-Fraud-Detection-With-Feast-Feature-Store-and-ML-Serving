"""
test_model.py
==============
Tests for model training and prediction logic.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import pytest
import os
import tempfile


# ── Sample data matching our feature schema ───────────────────────────────────
def get_sample_data(n_samples=200):
    """Generate sample data matching the feature schema."""
    np.random.seed(42)
    n_fraud = max(10, n_samples // 20)
    n_legit = n_samples - n_fraud

    feature_cols = [f"v{i}" for i in range(1, 29)] + [
        "scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"
    ]

    # Legit transactions
    data_legit = np.random.randn(n_legit, 28).astype(np.float32)
    # Fraud transactions (shifted distribution)
    data_fraud = np.random.randn(n_fraud, 28).astype(np.float32)
    data_fraud[:, 0] -= 2.5
    data_fraud[:, 13] -= 3.5

    data = np.vstack([data_legit, data_fraud])

    df = pd.DataFrame(data, columns=[f"v{i}" for i in range(1, 29)])
    df["scaled_amount"] = np.random.randn(n_samples).astype(np.float32)
    df["scaled_time"] = np.random.randn(n_samples).astype(np.float32)
    df["hour_of_day"] = np.random.randint(0, 24, n_samples)
    df["is_night"] = (df["hour_of_day"].isin([23, 0, 1, 2, 3, 4, 5])).astype(int)
    df["amount_zscore"] = np.random.randn(n_samples).astype(np.float32)
    df["is_fraud"] = [0] * n_legit + [1] * n_fraud

    return df, feature_cols


class TestModelTraining:
    """Tests for XGBoost model training."""

    def test_model_trains_without_error(self):
        """XGBoost must fit without raising any exception."""
        df, feature_cols = get_sample_data()
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)
        assert model is not None

    def test_predictions_are_binary(self):
        """Model must only predict 0 or 1."""
        df, feature_cols = get_sample_data()
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)
        predictions = model.predict(X)

        assert set(np.unique(predictions)).issubset({0, 1})

    def test_probabilities_between_0_and_1(self):
        """Predicted probabilities must be in [0, 1]."""
        df, feature_cols = get_sample_data()
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)
        probas = model.predict_proba(X)[:, 1]

        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_model_saves_and_loads(self, tmp_path):
        """Model saved with save_model must load back correctly."""
        df, feature_cols = get_sample_data()
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)

        model_path = str(tmp_path / "model.json")
        model.save_model(model_path)

        loaded = xgb.XGBClassifier()
        loaded.load_model(model_path)

        # Predictions should match
        orig_preds = model.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)

    def test_model_handles_class_imbalance(self):
        """Model with scale_pos_weight should detect some fraud."""
        df, feature_cols = get_sample_data(n_samples=500)
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=50, max_depth=4, scale_pos_weight=20,
            random_state=42, use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)
        predictions = model.predict(X)

        # Should predict at least some fraud
        assert predictions.sum() > 0

    def test_feature_importance_exists(self):
        """Model must have feature importances after training."""
        df, feature_cols = get_sample_data()
        X = df[feature_cols]
        y = df["is_fraud"]

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=3, random_state=42,
            use_label_encoder=False, eval_metric="logloss"
        )
        model.fit(X, y)

        importances = model.feature_importances_
        assert len(importances) == len(feature_cols)
        assert importances.sum() > 0
