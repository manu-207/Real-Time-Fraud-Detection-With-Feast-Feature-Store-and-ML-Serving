#!/bin/bash
# ── Run Full ML Pipeline ──────────────────────────────────────────────────────
# Executes all pipeline stages in order:
# data_prep → feature_engineering → feast setup → train → evaluate → monitor

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Credit Card Fraud Detection — Full Pipeline"
echo "═══════════════════════════════════════════════════════"
echo ""

# Step 1: Data preparation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/6] Data Preparation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/data_prep.py
echo ""

# Step 2: Feature engineering
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/6] Feature Engineering"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/feature_engineering.py
echo ""

# Step 3: Feast setup (requires Redis)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [3/6] Feast Feature Store Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if redis-cli ping > /dev/null 2>&1; then
    feast -c feature_repo apply
    feast -c feature_repo materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"
    echo "  ✓ Features materialized to Redis"
else
    echo "  ⚠ Redis not running — skipping Feast materialization"
    echo "    (Online serving via /predict/feast won't work)"
fi
echo ""

# Step 4: Model training
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [4/6] Model Training (XGBoost)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/train.py
echo ""

# Step 5: Evaluation
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [5/6] Model Evaluation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/evaluate.py
echo ""

# Step 6: Drift monitoring
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [6/6] Drift Monitoring (Evidently)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python src/monitor.py
echo ""

echo "═══════════════════════════════════════════════════════"
echo "  ✅ Pipeline complete!"
echo ""
echo "  Next steps:"
echo "    • Start API:  uvicorn serving.app:app --port 8000"
echo "    • View docs:  http://localhost:8000/docs"
echo "    • MLflow UI:  http://localhost:5000"
echo "    • Metrics:    http://localhost:8000/metrics"
echo "═══════════════════════════════════════════════════════"
