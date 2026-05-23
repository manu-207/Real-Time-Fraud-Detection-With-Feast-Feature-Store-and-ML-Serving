"""
schemas.py
===========
Pydantic models for request/response validation.
FastAPI uses these for automatic OpenAPI documentation and input validation.
"""

from pydantic import BaseModel, Field
from typing import Optional


class TransactionRequest(BaseModel):
    """Input schema for a single transaction prediction."""

    transaction_id: int = Field(..., description="Unique transaction identifier", example=12345)
    v1: float = Field(..., description="PCA feature V1", example=-1.36)
    v2: float = Field(..., description="PCA feature V2", example=-0.07)
    v3: float = Field(..., description="PCA feature V3", example=2.54)
    v4: float = Field(..., description="PCA feature V4", example=1.38)
    v5: float = Field(..., description="PCA feature V5", example=-0.34)
    v6: float = Field(..., description="PCA feature V6", example=0.46)
    v7: float = Field(..., description="PCA feature V7", example=0.24)
    v8: float = Field(..., description="PCA feature V8", example=-0.04)
    v9: float = Field(..., description="PCA feature V9", example=0.57)
    v10: float = Field(..., description="PCA feature V10", example=-0.38)
    v11: float = Field(..., description="PCA feature V11", example=-0.23)
    v12: float = Field(..., description="PCA feature V12", example=-0.04)
    v13: float = Field(..., description="PCA feature V13", example=-0.44)
    v14: float = Field(..., description="PCA feature V14", example=-0.12)
    v15: float = Field(..., description="PCA feature V15", example=-0.07)
    v16: float = Field(..., description="PCA feature V16", example=-0.23)
    v17: float = Field(..., description="PCA feature V17", example=-0.29)
    v18: float = Field(..., description="PCA feature V18", example=-0.11)
    v19: float = Field(..., description="PCA feature V19", example=0.01)
    v20: float = Field(..., description="PCA feature V20", example=-0.15)
    v21: float = Field(..., description="PCA feature V21", example=-0.07)
    v22: float = Field(..., description="PCA feature V22", example=-0.23)
    v23: float = Field(..., description="PCA feature V23", example=-0.07)
    v24: float = Field(..., description="PCA feature V24", example=0.56)
    v25: float = Field(..., description="PCA feature V25", example=-0.32)
    v26: float = Field(..., description="PCA feature V26", example=-0.09)
    v27: float = Field(..., description="PCA feature V27", example=-0.02)
    v28: float = Field(..., description="PCA feature V28", example=-0.05)
    scaled_amount: float = Field(..., description="StandardScaler(Amount)", example=0.24)
    scaled_time: float = Field(..., description="StandardScaler(Time)", example=-0.99)
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of transaction (0-23)", example=14)
    is_night: int = Field(..., ge=0, le=1, description="1 if night transaction (11pm-5am)", example=0)
    amount_zscore: float = Field(..., description="Z-score of transaction amount", example=0.15)

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": 12345,
                "v1": -1.36, "v2": -0.07, "v3": 2.54, "v4": 1.38,
                "v5": -0.34, "v6": 0.46, "v7": 0.24, "v8": -0.04,
                "v9": 0.57, "v10": -0.38, "v11": -0.23, "v12": -0.04,
                "v13": -0.44, "v14": -0.12, "v15": -0.07, "v16": -0.23,
                "v17": -0.29, "v18": -0.11, "v19": 0.01, "v20": -0.15,
                "v21": -0.07, "v22": -0.23, "v23": -0.07, "v24": 0.56,
                "v25": -0.32, "v26": -0.09, "v27": -0.02, "v28": -0.05,
                "scaled_amount": 0.24, "scaled_time": -0.99,
                "hour_of_day": 14, "is_night": 0, "amount_zscore": 0.15,
            }
        }


class TransactionFromFeast(BaseModel):
    """Lightweight request — only entity ID needed. Features fetched from Feast."""

    transaction_id: int = Field(..., description="Transaction ID to look up in feature store", example=12345)


class PredictionResponse(BaseModel):
    """Output schema for fraud prediction."""

    transaction_id: int = Field(..., description="Transaction identifier")
    is_fraud: bool = Field(..., description="True if predicted as fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk category: low/medium/high/critical")
    latency_ms: float = Field(..., description="End-to-end prediction latency in ms")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": 12345,
                "is_fraud": False,
                "fraud_probability": 0.023,
                "risk_level": "low",
                "latency_ms": 3.2,
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., example="healthy")
    model_loaded: bool
    feast_connected: bool
    redis_connected: bool


class DriftSummaryResponse(BaseModel):
    """Drift monitoring summary."""

    drift_share: float
    amount_drift: int
    prediction_drift: int
    predictions_since_last_check: int
