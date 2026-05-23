#!/bin/bash
# ── Setup Feast Feature Store ─────────────────────────────────────────────────
# Run this after feature_engineering.py to register features and materialize
# them to the Redis online store.

set -e

echo "═══════════════════════════════════════════════════════"
echo "  Setting up Feast Feature Store"
echo "═══════════════════════════════════════════════════════"

# Check Redis is running
echo "[1/4] Checking Redis connection..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "ERROR: Redis is not running. Start it with:"
    echo "  docker-compose up -d redis"
    exit 1
fi
echo "  ✓ Redis is running"

# Check feature data exists
echo "[2/4] Checking feature data..."
if [ ! -f "feature_repo/data/features.parquet" ]; then
    echo "ERROR: Feature data not found. Run feature engineering first:"
    echo "  python src/feature_engineering.py"
    exit 1
fi
echo "  ✓ Feature data found"

# Apply Feast definitions
echo "[3/4] Applying Feast feature definitions..."
feast -c feature_repo apply
echo "  ✓ Feature views registered"

# Materialize features to Redis
echo "[4/4] Materializing features to Redis online store..."
feast -c feature_repo materialize-incremental "$(date -u +"%Y-%m-%dT%H:%M:%S")"
echo "  ✓ Features materialized to Redis"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ Feast setup complete!"
echo "  Features are now available in Redis for online serving."
echo "═══════════════════════════════════════════════════════"
