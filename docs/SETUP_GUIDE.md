# 🛠️ Manual Setup Guide — Step by Step

Complete guide to run the Credit Card Fraud Detection project from scratch.

---

## Prerequisites

- Python 3.10+
- Docker installed
- (Optional) Kaggle account for real dataset

---

## Step 1: Clone & Setup Virtual Environment

```bash
# Navigate to project
cd credit-card-fraud-detection

# Create virtual environment
python -m venv venv

# Activate it
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Step 2: Start Redis (required for Feast online store)

```bash
# Start Redis using Docker
docker run -d --name redis -p 6379:6379 redis:7-alpine

# Verify Redis is running
docker exec redis redis-cli ping
# Should print: PONG
```

---

## Step 3: Start MLflow (experiment tracking)

```bash
# Create directory for MLflow data
mkdir -p mlflow_data

# Start MLflow server (run in a separate terminal)
mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow_data/mlflow.db \
    --default-artifact-root ./mlflow_data/artifacts

# Verify: open http://localhost:5000 in browser
```

---

## Step 4: Set Environment Variables

```bash
# Create .env file
cp .env.example .env

# Edit .env with these values for local development:
export REDIS_HOST=localhost
export REDIS_PORT=6379
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=fraud-detection
export MODEL_PATH=models/model.json
export FEAST_REPO_PATH=feature_repo
```

Or just export them directly in your terminal:
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=fraud-detection
```

---

## Step 5: Prepare Data

**Option A — Use synthetic data (no Kaggle needed):**
```bash
python src/data_prep.py
```
Output:
```
[data_prep] Generating synthetic credit card fraud dataset...
[data_prep] Dataset saved to data/raw/creditcard.csv
[data_prep] Shape: (284807, 31)
[data_prep] Fraud ratio: 0.1727%
```

**Option B — Use real Kaggle dataset:**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip -d data/raw/
mv data/raw/creditcard.csv data/raw/creditcard.csv
```

---

## Step 6: Feature Engineering

```bash
python src/feature_engineering.py
```

Output:
```
[features] Loading raw data from data/raw/creditcard.csv
[features] Raw shape: (284807, 31)
[features] Feature parquet saved to data/processed/features.parquet
[features] Feast source parquet saved to feature_repo/data/features.parquet
[features] Shape: (284807, 36)
[features] Dataset statistics:
  Total transactions: 284,807
  Fraud transactions: 492
  Fraud ratio: 0.1727%
```

This creates:
- `data/processed/features.parquet` — for training
- `feature_repo/data/features.parquet` — for Feast offline store

---

## Step 7: Setup Feast Feature Store

```bash
# Apply feature definitions (registers entity + feature views)
feast -c feature_repo apply
```

Output:
```
Applying changes for project fraud_detection
Created entity transaction_id
Created feature view transaction_features
```

```bash
# Materialize features to Redis (offline → online store)
feast -c feature_repo materialize-incremental "2024-01-03T00:00:00"
```

Output:
```
Materializing feature view transaction_features from 2024-01-01 to 2024-01-03
Done!
```

**Verify features are in Redis:**
```bash
docker exec redis redis-cli DBSIZE
# Should show a non-zero number (thousands of keys)
```

---

## Step 8: Train the Model

```bash
python src/train.py
```

Output:
```
[train] Loading features from data/processed/features.parquet
[train] Features: 33, Samples: 284,807
[train] Fraud ratio: 0.1727%
[train] Train: 227,845 | Test: 56,962
[0]     validation_0-aucpr:0.xxxxx
[20]    validation_0-aucpr:0.xxxxx
...
[train] Results:
  ROC-AUC: 0.98xx
  PR-AUC:  0.85xx
  F1:      0.8xxx

[train] Top 10 features:
  1. v14: 0.xxxx
  2. v17: 0.xxxx
  3. v12: 0.xxxx
  ...

[train] Model saved to models/model.json
```

**Check MLflow UI:** Open http://localhost:5000
- You'll see the experiment "fraud-detection"
- Click the run to see params, metrics, and artifacts

---

## Step 9: Evaluate the Model

```bash
python src/evaluate.py
```

Output:
```
[evaluate] Full Dataset Evaluation:
  Samples: 284,807
  ROC-AUC: 0.99xx
  PR-AUC:  0.87xx
  F1:      0.8xxx

[evaluate] Optimal threshold: 0.xxxx
[evaluate] Report saved to reports/evaluation_report.json
```

---

## Step 10: Run Drift Monitoring

```bash
python src/monitor.py
```

Output:
```
[monitor] Reference: 199,364 | Current: 85,443
[monitor] HTML report saved to reports/drift_report.html
[monitor] JSON summary saved to reports/drift_summary.json
[monitor] Drift metrics: {'drift_share': 0.0, ...}
[monitor] Metrics + artifacts logged to MLflow
```

---

## Step 11: Start the FastAPI Server

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Output:
```
[app] Model loaded successfully
[app] Feast store connected
[app] Redis connected
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## Step 12: Test the API

**Open Swagger UI:** http://localhost:8000/docs

**Test health endpoint:**
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "healthy",
  "model_loaded": true,
  "feast_connected": true,
  "redis_connected": true
}
```

**Test direct prediction:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": 1,
    "v1": -1.36, "v2": -0.07, "v3": 2.54, "v4": 1.38,
    "v5": -0.34, "v6": 0.46, "v7": 0.24, "v8": -0.04,
    "v9": 0.57, "v10": -0.38, "v11": -0.23, "v12": -0.04,
    "v13": -0.44, "v14": -0.12, "v15": -0.07, "v16": -0.23,
    "v17": -0.29, "v18": -0.11, "v19": 0.01, "v20": -0.15,
    "v21": -0.07, "v22": -0.23, "v23": -0.07, "v24": 0.56,
    "v25": -0.32, "v26": -0.09, "v27": -0.02, "v28": -0.05,
    "scaled_amount": 0.24, "scaled_time": -0.99,
    "hour_of_day": 14, "is_night": 0, "amount_zscore": 0.15
  }'
```
```json
{
  "transaction_id": 1,
  "is_fraud": false,
  "fraud_probability": 0.023,
  "risk_level": "low",
  "latency_ms": 2.8
}
```

**Test Feast prediction (features from Redis):**
```bash
curl -X POST http://localhost:8000/predict/feast \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": 100}'
```
```json
{
  "transaction_id": 100,
  "is_fraud": false,
  "fraud_probability": 0.012,
  "risk_level": "low",
  "latency_ms": 4.1
}
```

**Check Prometheus metrics:**
```bash
curl http://localhost:8000/metrics
```

**Check drift summary:**
```bash
curl http://localhost:8000/drift-summary
```

---

## Step 13: Run Tests

```bash
pytest tests/ -v
```

---

## Step 14: Load Test (trigger drift detection)

```bash
# Send 200 random predictions to trigger live drift check
bash scripts/send_predictions.sh http://localhost:8000/predict 200
```

After 100 predictions, the live drift check runs automatically.
Check: http://localhost:8000/drift-report

---

## 📋 Quick Reference — All Commands in Order

```bash
# 1. Setup
cd credit-card-fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Infrastructure
docker run -d --name redis -p 6379:6379 redis:7-alpine
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow_data/mlflow.db --default-artifact-root ./mlflow_data/artifacts &

# 3. Environment
export REDIS_HOST=localhost
export REDIS_PORT=6379
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=fraud-detection

# 4. Pipeline
python src/data_prep.py
python src/feature_engineering.py
feast -c feature_repo apply
feast -c feature_repo materialize-incremental "2024-01-03T00:00:00"
python src/train.py
python src/evaluate.py
python src/monitor.py

# 5. Serve
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload

# 6. Test
pytest tests/ -v
curl http://localhost:8000/health
```

---

## 🛑 Troubleshooting

| Problem | Solution |
|---|---|
| `redis.ConnectionError` | Make sure Redis container is running: `docker ps` |
| `feast: command not found` | `pip install feast[redis]` and check PATH |
| `MLflow connection refused` | Start MLflow server first (Step 3) |
| `Model not found` | Run `python src/train.py` first |
| `Features not found in online store` | Run `feast materialize-incremental` |
| `No module named 'serving'` | Run from project root directory |
| Port already in use | Kill existing process: `lsof -i :8000` then `kill <PID>` |

---

## 🧹 Cleanup

```bash
# Stop Redis
docker stop redis && docker rm redis

# Stop MLflow (if running in background)
pkill -f "mlflow server"

# Remove generated files
rm -rf data/raw data/processed models/ reports/ mlflow_data/
rm -rf feature_repo/data/features.parquet feature_repo/data/registry.db
```
