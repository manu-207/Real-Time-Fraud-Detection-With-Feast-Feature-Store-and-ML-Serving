"""
app.py — FastAPI Fraud Detection Serving Application
=====================================================
Real-time fraud prediction with:
- Direct prediction (all features in request body)
- Feature store prediction (only entity ID, features fetched from Feast/Redis)
- Prometheus metrics for observability
- Live Evidently drift monitoring
- Auto-generated OpenAPI docs at /docs

Endpoints:
    POST /predict          — predict with all features in request
    POST /predict/feast    — predict using Feast online store (Redis)
    GET  /health           — health check (model + Redis + Feast)
    GET  /metrics          — Prometheus scrape endpoint
    GET  /drift-summary    — latest drift metrics
    GET  /docs             — Swagger UI (auto-generated)
"""

import time
import os
import json
import threading
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from serving.schemas import (
    TransactionRequest,
    TransactionFromFeast,
    PredictionResponse,
    HealthResponse,
    DriftSummaryResponse,
)
from serving.dependencies import (
    load_model,
    get_feast_store,
    get_redis_client,
    get_feature_columns,
    get_risk_level,
    REDIS_HOST,
    REDIS_PORT,
)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
PREDICTION_COUNTER = Counter(
    "fraud_predictions_total", "Total predictions", ["result", "risk_level"]
)
PREDICTION_LATENCY = Histogram(
    "fraud_prediction_latency_seconds", "Prediction latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)
FEAST_LATENCY = Histogram(
    "fraud_feast_latency_seconds", "Feast online store fetch latency",
    buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05]
)
MODEL_LOADED_GAUGE = Gauge("fraud_model_loaded", "1 if model is loaded")
DRIFT_SHARE_GAUGE = Gauge("fraud_drift_share", "Share of drifted features")
PREDICTIONS_BUFFER_SIZE = Gauge("fraud_predictions_buffer_size", "Predictions in drift buffer")

# ── Global State ──────────────────────────────────────────────────────────────
model = None
feast_store = None
redis_client = None
prediction_buffer = []
drift_lock = threading.Lock()
DRIFT_CHECK_EVERY = int(os.getenv("DRIFT_CHECK_EVERY", "100"))


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect to services on startup."""
    global model, feast_store, redis_client

    # Load model
    try:
        model = load_model()
        MODEL_LOADED_GAUGE.set(1)
        print("[app] Model loaded successfully")
    except Exception as e:
        print(f"[app] WARNING: Could not load model: {e}")
        MODEL_LOADED_GAUGE.set(0)

    # Connect to Feast
    feast_store = get_feast_store()
    if feast_store:
        print("[app] Feast store connected")

    # Connect to Redis
    redis_client = get_redis_client()
    if redis_client:
        print("[app] Redis connected")

    yield

    # Cleanup
    print("[app] Shutting down...")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time fraud detection using XGBoost + Feast Feature Store + Redis. "
        "Features are served from Redis at <5ms latency for production transactions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if all services are healthy."""
    redis_ok = False
    if redis_client:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            pass

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        feast_connected=feast_store is not None,
        redis_connected=redis_ok,
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_direct(request: TransactionRequest):
    """
    Predict fraud with all features provided in the request body.
    Use this when features are computed client-side.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()

    # Extract features in correct order
    feature_cols = get_feature_columns()
    features = np.array([[getattr(request, col) for col in feature_cols]])

    # Predict
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])
    risk_level = get_risk_level(probability)

    latency_ms = (time.time() - start) * 1000

    # Record metrics
    PREDICTION_LATENCY.observe(time.time() - start)
    PREDICTION_COUNTER.labels(
        result="fraud" if prediction == 1 else "legit",
        risk_level=risk_level,
    ).inc()

    # Buffer for drift detection
    _log_prediction(request, probability)

    return PredictionResponse(
        transaction_id=request.transaction_id,
        is_fraud=prediction == 1,
        fraud_probability=round(probability, 4),
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/feast", response_model=PredictionResponse, tags=["Prediction"])
async def predict_from_feast(request: TransactionFromFeast):
    """
    Predict fraud using features from Feast online store (Redis).
    Only the transaction_id is needed — features are fetched at <5ms.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if feast_store is None:
        raise HTTPException(status_code=503, detail="Feast store not connected")

    start = time.time()

    # Fetch features from Feast online store (Redis)
    feast_start = time.time()
    try:
        feature_vector = feast_store.get_online_features(
            features=[
                f"transaction_features:{col}" for col in get_feature_columns()
            ],
            entity_rows=[{"transaction_id": request.transaction_id}],
        ).to_dict()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Features not found for transaction {request.transaction_id}: {e}")

    feast_latency = time.time() - feast_start
    FEAST_LATENCY.observe(feast_latency)

    # Build feature array
    feature_cols = get_feature_columns()
    features = np.array([[feature_vector[col][0] for col in feature_cols]])

    # Check for None values (entity not in online store)
    if any(v is None for v in features[0]):
        raise HTTPException(
            status_code=404,
            detail=f"Transaction {request.transaction_id} not found in online store"
        )

    # Predict
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0, 1])
    risk_level = get_risk_level(probability)

    latency_ms = (time.time() - start) * 1000

    # Record metrics
    PREDICTION_LATENCY.observe(time.time() - start)
    PREDICTION_COUNTER.labels(
        result="fraud" if prediction == 1 else "legit",
        risk_level=risk_level,
    ).inc()

    return PredictionResponse(
        transaction_id=request.transaction_id,
        is_fraud=prediction == 1,
        fraud_probability=round(probability, 4),
        risk_level=risk_level,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/drift-summary", response_model=DriftSummaryResponse, tags=["Monitoring"])
async def drift_summary():
    """Get the latest drift monitoring summary."""
    json_path = "reports/drift_summary.json"
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        return DriftSummaryResponse(
            drift_share=data.get("drift_share", 0),
            amount_drift=data.get("amount_drift", 0),
            prediction_drift=data.get("prediction_drift", 0),
            predictions_since_last_check=len(prediction_buffer),
        )
    return DriftSummaryResponse(
        drift_share=0, amount_drift=0, prediction_drift=0,
        predictions_since_last_check=len(prediction_buffer),
    )


@app.get("/drift-report", response_class=HTMLResponse, tags=["Monitoring"])
async def drift_report():
    """Serve the latest Evidently HTML drift report."""
    for path in ["reports/live_drift_report.html", "reports/drift_report.html"]:
        if os.path.exists(path):
            with open(path) as f:
                return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="<h2>No drift report available yet. Run the monitoring pipeline first.</h2>",
        status_code=404,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _log_prediction(request: TransactionRequest, probability: float):
    """Buffer predictions for periodic drift checks."""
    global prediction_buffer

    feature_cols = get_feature_columns()
    row = {col: getattr(request, col) for col in feature_cols}
    row["prediction_proba"] = probability

    with drift_lock:
        prediction_buffer.append(row)
        count = len(prediction_buffer)

    PREDICTIONS_BUFFER_SIZE.set(count)

    if count % DRIFT_CHECK_EVERY == 0:
        t = threading.Thread(target=_run_live_drift_check, daemon=True)
        t.start()


def _run_live_drift_check():
    """Background drift check using Evidently."""
    global prediction_buffer

    try:
        from evidently.report import Report
        from evidently.metrics import DatasetDriftMetric

        ref_path = os.getenv("REFERENCE_DATA_PATH", "data/processed/features.parquet")
        if not os.path.exists(ref_path):
            return

        ref_df = pd.read_parquet(ref_path)
        feature_cols = get_feature_columns()

        with drift_lock:
            current_data = prediction_buffer[-DRIFT_CHECK_EVERY:]

        cur_df = pd.DataFrame(current_data)

        # Only check overlapping columns
        check_cols = [c for c in feature_cols if c in cur_df.columns and c in ref_df.columns]

        report = Report(metrics=[DatasetDriftMetric()])
        report.run(reference_data=ref_df[check_cols], current_data=cur_df[check_cols])

        result = report.as_dict()
        metrics = result.get("metrics", [])
        for m in metrics:
            if "DatasetDriftMetric" in str(m.get("metric", "")):
                share = m.get("result", {}).get("share_of_drifted_columns", 0)
                DRIFT_SHARE_GAUGE.set(share)

        # Save live report
        os.makedirs("reports", exist_ok=True)
        report.save_html("reports/live_drift_report.html")
        print(f"[drift] Live check complete. Buffer size: {len(prediction_buffer)}")

    except Exception as e:
        print(f"[drift] Error in live drift check: {e}")
