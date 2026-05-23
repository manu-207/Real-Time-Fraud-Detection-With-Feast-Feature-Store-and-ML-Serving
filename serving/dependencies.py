"""
dependencies.py
================
Shared dependencies for the FastAPI app: model loading, Feast store,
and Redis connection management.
"""

import os
import json
import xgboost as xgb
from feast import FeatureStore
import redis
import yaml


# ── Configuration ─────────────────────────────────────────────────────────────
FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_repo")
MODEL_PATH = os.getenv("MODEL_PATH", "models/model.json")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


def get_feature_columns():
    """Return the ordered list of feature columns the model expects."""
    v_cols = [f"v{i}" for i in range(1, 29)]
    engineered = ["scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"]
    return v_cols + engineered


def load_model():
    """Load XGBoost model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def get_feast_store():
    """Initialize Feast feature store connection."""
    try:
        store = FeatureStore(repo_path=FEAST_REPO_PATH)
        return store
    except Exception as e:
        print(f"[dependencies] WARNING: Could not connect to Feast: {e}")
        return None


def get_redis_client():
    """Get Redis client for health checks."""
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        return client
    except Exception as e:
        print(f"[dependencies] WARNING: Could not connect to Redis: {e}")
        return None


def get_risk_level(probability: float) -> str:
    """Classify fraud probability into risk levels."""
    if probability < 0.1:
        return "low"
    elif probability < 0.5:
        return "medium"
    elif probability < 0.8:
        return "high"
    else:
        return "critical"
