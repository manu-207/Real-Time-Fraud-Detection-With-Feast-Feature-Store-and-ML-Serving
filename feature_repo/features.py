"""
Feast Feature Definitions for Credit Card Fraud Detection
==========================================================
Defines the entity, data source, and feature views that Feast uses to
serve features both offline (for training) and online (for real-time inference).
"""

from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

import os

# Use absolute path to avoid Feast's inconsistent relative path resolution
_REPO_DIR = os.path.dirname(os.path.abspath(__file__))
_PARQUET_PATH = os.path.join(_REPO_DIR, "data", "features.parquet")

# ── Entity: each transaction has a unique ID ──────────────────────────────────
transaction = Entity(
    name="transaction_id",
    description="Unique identifier for each credit card transaction",
)

# ── Data Source: parquet file with pre-computed features ──────────────────────
transaction_source = FileSource(
    path=_PARQUET_PATH,
    timestamp_field="event_timestamp",
)

# ── Feature View: transaction features ────────────────────────────────────────
# These features are computed in src/feature_engineering.py and materialized
# to Redis for online serving at <5ms latency.
transaction_features = FeatureView(
    name="transaction_features",
    entities=[transaction],
    ttl=timedelta(days=7),  # features expire after 7 days in online store
    schema=[
        # Original PCA-transformed features (V1-V28 from the dataset)
        Field(name="v1", dtype=Float32),
        Field(name="v2", dtype=Float32),
        Field(name="v3", dtype=Float32),
        Field(name="v4", dtype=Float32),
        Field(name="v5", dtype=Float32),
        Field(name="v6", dtype=Float32),
        Field(name="v7", dtype=Float32),
        Field(name="v8", dtype=Float32),
        Field(name="v9", dtype=Float32),
        Field(name="v10", dtype=Float32),
        Field(name="v11", dtype=Float32),
        Field(name="v12", dtype=Float32),
        Field(name="v13", dtype=Float32),
        Field(name="v14", dtype=Float32),
        Field(name="v15", dtype=Float32),
        Field(name="v16", dtype=Float32),
        Field(name="v17", dtype=Float32),
        Field(name="v18", dtype=Float32),
        Field(name="v19", dtype=Float32),
        Field(name="v20", dtype=Float32),
        Field(name="v21", dtype=Float32),
        Field(name="v22", dtype=Float32),
        Field(name="v23", dtype=Float32),
        Field(name="v24", dtype=Float32),
        Field(name="v25", dtype=Float32),
        Field(name="v26", dtype=Float32),
        Field(name="v27", dtype=Float32),
        Field(name="v28", dtype=Float32),
        # Scaled amount and time-based features
        Field(name="scaled_amount", dtype=Float32),
        Field(name="scaled_time", dtype=Float32),
        # Engineered features
        Field(name="hour_of_day", dtype=Int64),
        Field(name="is_night", dtype=Int64),  # 1 if transaction between 11pm-5am
        Field(name="amount_zscore", dtype=Float32),
    ],
    source=transaction_source,
    online=True,  # materialize to Redis for real-time serving
)
