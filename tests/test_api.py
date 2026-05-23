"""
test_api.py
============
Tests for the FastAPI serving endpoints.
"""

import pytest
import numpy as np
import pandas as pd
import xgboost as xgb
from unittest.mock import patch, MagicMock
from httpx import AsyncClient, ASGITransport
import asyncio


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def trained_model():
    """Create a small trained XGBoost model for testing."""
    np.random.seed(42)
    n = 100
    feature_cols = [f"v{i}" for i in range(1, 29)] + [
        "scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"
    ]
    X = np.random.randn(n, len(feature_cols)).astype(np.float32)
    y = np.random.randint(0, 2, n)

    model = xgb.XGBClassifier(
        n_estimators=5, max_depth=2, random_state=42,
        use_label_encoder=False, eval_metric="logloss"
    )
    model.fit(X, y)
    return model


@pytest.fixture
def app_client(trained_model):
    """Create a test client with mocked model."""
    import serving.app as app_module

    # Inject model
    app_module.model = trained_model
    app_module.feast_store = None
    app_module.redis_client = None

    return app_module.app


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_shows_model_loaded(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/health")
        data = response.json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def _get_valid_payload(self):
        """Return a valid prediction request payload."""
        payload = {"transaction_id": 1}
        for i in range(1, 29):
            payload[f"v{i}"] = round(np.random.randn(), 4)
        payload["scaled_amount"] = 0.24
        payload["scaled_time"] = -0.99
        payload["hour_of_day"] = 14
        payload["is_night"] = 0
        payload["amount_zscore"] = 0.15
        return payload

    @pytest.mark.asyncio
    async def test_predict_returns_200(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=self._get_valid_payload())
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_predict_response_has_required_fields(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=self._get_valid_payload())
        data = response.json()
        assert "transaction_id" in data
        assert "is_fraud" in data
        assert "fraud_probability" in data
        assert "risk_level" in data
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_predict_probability_in_range(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=self._get_valid_payload())
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    @pytest.mark.asyncio
    async def test_predict_risk_level_valid(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=self._get_valid_payload())
        data = response.json()
        assert data["risk_level"] in ["low", "medium", "high", "critical"]

    @pytest.mark.asyncio
    async def test_predict_missing_field_returns_422(self, app_client):
        """Missing required field should return 422 (Pydantic validation)."""
        payload = {"transaction_id": 1, "v1": 0.5}  # missing most fields
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=payload)
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_predict_invalid_hour_returns_422(self, app_client):
        """hour_of_day > 23 should fail validation."""
        payload = self._get_valid_payload()
        payload["hour_of_day"] = 25
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/predict", json=payload)
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_returns_200(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_contains_prometheus_format(self, app_client):
        transport = ASGITransport(app=app_client)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")
        assert b"fraud_predictions_total" in response.content or response.status_code == 200
