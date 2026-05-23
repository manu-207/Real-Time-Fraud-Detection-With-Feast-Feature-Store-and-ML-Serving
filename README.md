# 💳 Real-Time Feature Store + ML Serving Pipeline — Credit Card Fraud Detection

A production-grade MLOps pipeline for real-time credit card fraud detection using Feast feature store, Redis online serving, XGBoost, and FastAPI. The system demonstrates end-to-end ML engineering: offline feature computation, online feature serving at <5ms latency, model training with experiment tracking, real-time inference, and drift monitoring.

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE                                 │
│                                                                          │
│  Raw Data → Feature Engineering → Feast Offline Store → XGBoost Train   │
│                                          │                    │          │
│                                          │              MLflow Log       │
│                                          ▼                    │          │
│                                   Feast Materialize           ▼          │
│                                          │            Model Registry     │
│                                          ▼                               │
│                                    Redis (Online)                         │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                        SERVING PIPELINE                                   │
│                                                                          │
│  Client Request → FastAPI → Feast Online Store (Redis) → XGBoost Predict│
│                      │              <5ms                       │          │
│                      │                                        ▼          │
│                      │                                  Response JSON     │
│                      ▼                                                   │
│               Prometheus Metrics + Evidently Drift Monitoring            │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Tool | Why |
|---|---|---|
| Feature Store | Feast + Redis | Industry-standard; offline training + online serving from Redis automatically |
| Online Store | Redis | Pre-computed features served in <5ms for real transactions |
| API Framework | FastAPI | Async endpoints, Pydantic validation, auto OpenAPI docs |
| Model | XGBoost | Better than RandomForest on tabular fraud data |
| Experiment Tracking | MLflow | Track feature importance, model versions, evaluation metrics |
| Drift Monitoring | Evidently | Feature drift detection on live predictions |
| Containerization | Docker + Docker Compose | Local dev + production deployment |
| Deployment | AWS ECS (Fargate) | Serverless container orchestration |

---

## 📁 Project Structure

```
credit-card-fraud-detection/
├── feature_repo/                # Feast feature repository
│   ├── feature_store.yaml      # Feast configuration (Redis online store)
│   ├── features.py             # Feature view definitions
│   └── data/                   # Offline parquet data source
├── src/
│   ├── data_prep.py            # Download + prepare credit card dataset
│   ├── feature_engineering.py  # Compute features → write to offline store
│   ├── train.py                # XGBoost training + MLflow logging
│   ├── evaluate.py             # Model evaluation + metrics
│   └── monitor.py              # Evidently drift detection
├── serving/
│   ├── app.py                  # FastAPI serving app
│   ├── schemas.py              # Pydantic request/response models
│   └── dependencies.py         # Model + Feast store loading
├── tests/
│   ├── test_features.py        # Feature store tests
│   ├── test_model.py           # Model training/prediction tests
│   └── test_api.py             # FastAPI endpoint tests
├── scripts/
│   ├── setup_feast.sh          # Initialize Feast + materialize features
│   ├── send_predictions.sh     # Load test script
│   └── run_pipeline.sh         # End-to-end pipeline runner
├── docker-compose.yml          # Redis + API + MLflow
├── Dockerfile                  # Production API image
├── requirements.txt            # Python dependencies
├── params.yaml                 # Hyperparameters and config
├── Makefile                    # Common commands
└── .github/
    └── workflows/
        └── ci.yml              # CI/CD pipeline
```

---

## ⚙️ Setup & Local Run

### Prerequisites

- Python 3.10+
- Docker & Docker Compose
- Redis (via Docker or local install)

### 1 — Clone and install

```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2 — Start infrastructure

```bash
docker-compose up -d redis mlflow
```

### 3 — Run the full pipeline

```bash
make pipeline
# Or step by step:
python src/data_prep.py
python src/feature_engineering.py
feast -c feature_repo apply
feast -c feature_repo materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
python src/train.py
python src/evaluate.py
```

### 4 — Start the API

```bash
uvicorn serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

---

## 🔑 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `REDIS_HOST` | `localhost` | Redis host for Feast online store |
| `REDIS_PORT` | `6379` | Redis port |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow server URL |
| `MLFLOW_EXPERIMENT_NAME` | `fraud-detection` | MLflow experiment |
| `MODEL_PATH` | `models/model.json` | Path to trained XGBoost model |
| `FEAST_REPO_PATH` | `feature_repo` | Path to Feast feature repo |

---

## 📄 License

MIT License
