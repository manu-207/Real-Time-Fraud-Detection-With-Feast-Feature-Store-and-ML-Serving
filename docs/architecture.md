# 💳 Credit Card Fraud Detection — Full Architecture

## High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                          │
│                          CREDIT CARD FRAUD DETECTION SYSTEM                               │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  ZONE 1: DATA & FEATURE PIPELINE                                                         │
│                                                                                          │
│  ┌──────────────┐    ┌────────────────────┐    ┌─────────────────────┐                   │
│  │              │    │                    │    │                     │                   │
│  │  Raw Data    │───▶│  Feature           │───▶│  Feast Offline      │                   │
│  │  (CSV/S3)   │    │  Engineering       │    │  Store (Parquet)    │                   │
│  │              │    │                    │    │                     │                   │
│  └──────────────┘    └────────────────────┘    └──────────┬──────────┘                   │
│                                                           │                              │
│                       src/data_prep.py                     │  feast materialize           │
│                       src/feature_engineering.py           │                              │
│                                                           ▼                              │
│                                                ┌─────────────────────┐                   │
│                                                │                     │                   │
│                                                │  Redis Online Store │                   │
│                                                │  (Port 6379)        │                   │
│                                                │  < 5ms latency      │                   │
│                                                │                     │                   │
│                                                └─────────────────────┘                   │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  ZONE 2: MODEL TRAINING & EXPERIMENT TRACKING                                            │
│                                                                                          │
│  ┌─────────────────────┐    ┌──────────────────┐    ┌─────────────────────────┐          │
│  │                     │    │                  │    │                         │          │
│  │  Feast Offline      │───▶│  XGBoost         │───▶│  MLflow                 │          │
│  │  Store (Parquet)    │    │  Training        │    │  (Port 5000)            │          │
│  │                     │    │                  │    │                         │          │
│  │  • V1-V28 (PCA)    │    │  • GridSearch    │    │  • Params logged        │          │
│  │  • scaled_amount   │    │  • Early Stop    │    │  • Metrics (AUC, F1)    │          │
│  │  • scaled_time     │    │  • scale_pos_wt  │    │  • Model registered     │          │
│  │  • hour_of_day     │    │                  │    │  • Artifacts stored     │          │
│  │  • is_night        │    └──────────────────┘    │                         │          │
│  │  • amount_zscore   │              │             └─────────────────────────┘          │
│  │                     │              │                                                  │
│  └─────────────────────┘              ▼                                                  │
│                              ┌──────────────────┐                                        │
│                              │  models/         │                                        │
│                              │  model.json      │                                        │
│                              │  model_meta.json │                                        │
│                              └──────────────────┘                                        │
│                                                                                          │
│                       src/train.py                                                        │
│                       src/evaluate.py                                                     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  ZONE 3: REAL-TIME SERVING (FastAPI + Feast + Redis)                                     │
│                                                                                          │
│                                                                                          │
│  ┌───────────┐     ┌──────────────────────────────────────────────────────┐              │
│  │           │     │  FastAPI Application (Port 8000)                     │              │
│  │  Client   │────▶│                                                      │              │
│  │  Request  │     │  POST /predict         POST /predict/feast           │              │
│  │           │     │  (all features)        (entity ID only)              │              │
│  └───────────┘     │       │                       │                      │              │
│                    │       │                       │                      │              │
│                    │       ▼                       ▼                      │              │
│                    │  ┌──────────┐    ┌────────────────────────┐          │              │
│                    │  │ XGBoost  │    │  Feast SDK             │          │              │
│                    │  │ Predict  │    │  get_online_features() │          │              │
│                    │  └──────────┘    └───────────┬────────────┘          │              │
│                    │       │                      │                       │              │
│                    │       │                      ▼                       │              │
│                    │       │          ┌─────────────────────┐            │              │
│                    │       │          │  Redis Online Store  │            │              │
│                    │       │          │  < 5ms response      │            │              │
│                    │       │          └─────────────────────┘            │              │
│                    │       │                      │                       │              │
│                    │       ▼                      ▼                       │              │
│                    │  ┌──────────────────────────────────┐               │              │
│                    │  │  XGBoost Predict                  │               │              │
│                    │  │  → is_fraud (bool)                │               │              │
│                    │  │  → fraud_probability (0-1)        │               │              │
│                    │  │  → risk_level (low/med/high/crit) │               │              │
│                    │  │  → latency_ms                     │               │              │
│                    │  └──────────────────────────────────┘               │              │
│                    │                                                      │              │
│                    └──────────────────────────────────────────────────────┘              │
│                                                                                          │
│                    serving/app.py                                                         │
│                    serving/schemas.py                                                     │
│                    serving/dependencies.py                                                │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  ZONE 4: MONITORING & OBSERVABILITY                                                      │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │  Prometheus Metrics (GET /metrics)                                               │     │
│  │                                                                                  │     │
│  │  • fraud_predictions_total{result, risk_level}    — Counter                      │     │
│  │  • fraud_prediction_latency_seconds               — Histogram                    │     │
│  │  • fraud_feast_latency_seconds                    — Histogram                    │     │
│  │  • fraud_model_loaded                             — Gauge                        │     │
│  │  • fraud_drift_share                              — Gauge                        │     │
│  │  • fraud_predictions_buffer_size                  — Gauge                        │     │
│  │                                                                                  │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │  Evidently AI Drift Detection                                                    │     │
│  │                                                                                  │     │
│  │  OFFLINE (src/monitor.py):                                                       │     │
│  │    • Runs after each training pipeline                                           │     │
│  │    • Compares reference (70%) vs current (30%) split                             │     │
│  │    • Generates reports/drift_report.html                                         │     │
│  │    • Generates reports/drift_summary.json                                        │     │
│  │    • Logs metrics to MLflow                                                      │     │
│  │                                                                                  │     │
│  │  LIVE (serving/app.py):                                                          │     │
│  │    • Buffers incoming predictions                                                │     │
│  │    • Every 100 predictions → runs Evidently drift check                         │     │
│  │    • Pushes drift_share to Prometheus gauge                                      │     │
│  │    • Saves live_drift_report.html                                                │     │
│  │                                                                                  │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────┐
│  ZONE 5: CI/CD & DEPLOYMENT                                                              │
│                                                                                          │
│  ┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐     │
│  │            │    │              │    │              │    │                      │     │
│  │  GitHub    │───▶│  GitHub      │───▶│  Docker      │───▶│  AWS ECR             │     │
│  │  Push      │    │  Actions     │    │  Build       │    │  (Container Registry)│     │
│  │            │    │              │    │              │    │                      │     │
│  └────────────┘    └──────────────┘    └──────────────┘    └──────────┬───────────┘     │
│                          │                                            │                  │
│                          │ pytest                                     │                  │
│                          │ train                                      ▼                  │
│                          │ evaluate                        ┌──────────────────────┐     │
│                          ▼                                 │                      │     │
│                    ┌──────────────┐                        │  AWS ECS Fargate     │     │
│                    │  Tests Pass  │                        │  (Container Service) │     │
│                    │  Model OK    │                        │                      │     │
│                    └──────────────┘                        │  • FastAPI app       │     │
│                                                           │  • Auto-scaling      │     │
│                                                           │  • Health checks     │     │
│                                                           │                      │     │
│                                                           └──────────────────────┘     │
│                                                                                          │
│  .github/workflows/ci.yml                                                                │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Infrastructure Diagram (AWS)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  AWS Cloud                                                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  VPC                                                                 │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  EC2 Instance (MLOps Server)                                  │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │    │
│  │  │  │   Redis     │  │   MLflow    │  │   Prometheus        │  │   │    │
│  │  │  │   :6379     │  │   :5000     │  │   :9090             │  │   │    │
│  │  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌─────────────────────┐                                     │   │    │
│  │  │  │   Grafana :3000     │                                     │   │    │
│  │  │  └─────────────────────┘                                     │   │    │
│  │  │                                                               │   │    │
│  │  └──────────────────────────────────────────────────────────────┘   │    │
│  │                                                                      │    │
│  │  ┌──────────────────────────────────────────────────────────────┐   │    │
│  │  │  ECS Fargate Cluster                                          │   │    │
│  │  │                                                               │   │    │
│  │  │  ┌─────────────────────────────────────────────────────────┐ │   │    │
│  │  │  │  Service: fraud-detection                                │ │   │    │
│  │  │  │                                                          │ │   │    │
│  │  │  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │ │   │    │
│  │  │  │  │  Task 1      │  │  Task 2      │  │  Task N      │  │ │   │    │
│  │  │  │  │  FastAPI     │  │  FastAPI     │  │  FastAPI     │  │ │   │    │
│  │  │  │  │  :8000       │  │  :8000       │  │  :8000       │  │ │   │    │
│  │  │  │  └──────────────┘  └──────────────┘  └──────────────┘  │ │   │    │
│  │  │  │                                                          │ │   │    │
│  │  │  └─────────────────────────────────────────────────────────┘ │   │    │
│  │  │                              ▲                                │   │    │
│  │  └──────────────────────────────┼────────────────────────────────┘   │    │
│  │                                 │                                     │    │
│  │  ┌──────────────────────────────┼────────────────────────────────┐   │    │
│  │  │  Application Load Balancer   │                                 │   │    │
│  │  │  (ALB) :443 / :80           │                                 │   │    │
│  │  └──────────────────────────────┼────────────────────────────────┘   │    │
│  │                                 │                                     │    │
│  └─────────────────────────────────┼─────────────────────────────────────┘    │
│                                    │                                          │
│  ┌─────────────────┐              │                                          │
│  │  AWS ECR        │              │                                          │
│  │  (Docker Images)│              │                                          │
│  └─────────────────┘              │                                          │
│                                    │                                          │
└────────────────────────────────────┼──────────────────────────────────────────┘
                                     │
                                     │  HTTPS
                                     │
                              ┌──────┴──────┐
                              │   Client    │
                              │  (Browser / │
                              │   API call) │
                              └─────────────┘
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  TRAINING FLOW (Batch — runs weekly or on-demand)                            │
│                                                                              │
│  ┌─────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐  │
│  │ Raw CSV │──▶│ Feature Eng. │──▶│ Parquet File │──▶│ Feast Apply      │  │
│  │ 284K    │   │ 33 features  │   │ (Offline)    │   │ + Materialize    │  │
│  │ rows    │   │ computed     │   │              │   │                  │  │
│  └─────────┘   └──────────────┘   └──────┬───────┘   └────────┬─────────┘  │
│                                           │                     │            │
│                                           ▼                     ▼            │
│                                    ┌──────────────┐   ┌──────────────────┐  │
│                                    │ XGBoost      │   │ Redis            │  │
│                                    │ Training     │   │ (Online Store)   │  │
│                                    │              │   │ 284K entities    │  │
│                                    └──────┬───────┘   └──────────────────┘  │
│                                           │                                  │
│                                           ▼                                  │
│                                    ┌──────────────┐                          │
│                                    │ model.json   │                          │
│                                    │ + MLflow log │                          │
│                                    └──────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  INFERENCE FLOW (Real-time — per transaction)                                │
│                                                                              │
│  Option A: Direct (all features in request)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Client ──▶ POST /predict {v1..v28, amount, time, ...}              │    │
│  │                    │                                                  │    │
│  │                    ▼                                                  │    │
│  │             Pydantic Validation                                       │    │
│  │                    │                                                  │    │
│  │                    ▼                                                  │    │
│  │             XGBoost.predict()  ──▶  Response JSON                    │    │
│  │                                     {is_fraud, probability, risk}    │    │
│  │                                                                      │    │
│  │  Latency: ~2-5ms                                                     │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Option B: Feast (only entity ID, features from Redis)                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │  Client ──▶ POST /predict/feast {transaction_id: 12345}             │    │
│  │                    │                                                  │    │
│  │                    ▼                                                  │    │
│  │             Feast SDK ──▶ Redis GET (33 features)  [< 3ms]          │    │
│  │                    │                                                  │    │
│  │                    ▼                                                  │    │
│  │             XGBoost.predict()  ──▶  Response JSON                    │    │
│  │                                     {is_fraud, probability, risk}    │    │
│  │                                                                      │    │
│  │  Latency: ~3-8ms (includes Redis round-trip)                         │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  MONITORING FLOW (Continuous)                                                │
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────────────┐     │
│  │ Every 100    │───▶│ Evidently Report │───▶│ Prometheus Gauges      │     │
│  │ predictions  │    │ (background      │    │ • drift_share          │     │
│  │ buffered     │    │  thread)         │    │ • amount_drift         │     │
│  └──────────────┘    └──────────────────┘    │ • prediction_drift     │     │
│                                              └───────────┬────────────┘     │
│                                                          │                   │
│                                                          ▼                   │
│                                              ┌────────────────────────┐     │
│                                              │ Grafana Dashboard      │     │
│                                              │ • Prediction rate      │     │
│                                              │ • Latency P50/P99      │     │
│                                              │ • Fraud ratio          │     │
│                                              │ • Drift alerts         │     │
│                                              └────────────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Matrix

```
┌────────────────────┬──────────┬───────┬─────────┬────────┬───────────┬──────────┐
│                    │ FastAPI  │ Redis │ MLflow  │ Feast  │ Evidently │ XGBoost  │
├────────────────────┼──────────┼───────┼─────────┼────────┼───────────┼──────────┤
│ FastAPI            │    —     │  R/W  │    —    │  Read  │   Read    │  Predict │
│ Redis              │  Serve   │   —   │    —    │ Store  │     —     │    —     │
│ MLflow             │    —     │   —   │    —    │   —    │   Log     │   Log    │
│ Feast              │  Serve   │  R/W  │    —    │   —    │     —     │    —     │
│ Evidently          │  Report  │   —   │   Log   │   —    │     —     │    —     │
│ XGBoost            │  Predict │   —   │   Log   │   —    │     —     │    —     │
└────────────────────┴──────────┴───────┴─────────┴────────┴───────────┴──────────┘
```

---

## API Endpoints

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FastAPI Endpoints (Port 8000)                                               │
├─────────────┬──────────────────┬────────────────────────────────────────────┤
│ Method      │ Path             │ Description                                │
├─────────────┼──────────────────┼────────────────────────────────────────────┤
│ POST        │ /predict         │ Predict with all features in body          │
│ POST        │ /predict/feast   │ Predict using Feast (entity ID only)       │
│ GET         │ /health          │ Health check (model + Redis + Feast)       │
│ GET         │ /metrics         │ Prometheus scrape endpoint                 │
│ GET         │ /drift-summary   │ Latest drift metrics (JSON)                │
│ GET         │ /drift-report    │ Evidently HTML drift report                │
│ GET         │ /docs            │ Swagger UI (auto-generated)                │
│ GET         │ /redoc           │ ReDoc API documentation                    │
└─────────────┴──────────────────┴────────────────────────────────────────────┘
```

---

## Pipeline Stages

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌────────┐
│  Data   │───▶│ Feature  │───▶│  Feast  │───▶│  Train   │───▶│Evaluate │───▶│Monitor │
│  Prep   │    │  Eng.    │    │  Apply  │    │ XGBoost  │    │ Metrics │    │ Drift  │
│         │    │          │    │  + Mat. │    │          │    │         │    │        │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘    └────────┘
     │              │               │               │               │             │
     ▼              ▼               ▼               ▼               ▼             ▼
 data/raw/    data/processed/   Redis         models/        reports/       reports/
 creditcard   features.parquet  (online)      model.json     eval.json      drift.html
 .csv                                         meta.json                     drift.json
```

---

## Security & Network

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Security Group Rules                                                        │
├──────────────┬──────────┬─────────────┬─────────────────────────────────────┤
│ Port         │ Protocol │ Source      │ Purpose                             │
├──────────────┼──────────┼─────────────┼─────────────────────────────────────┤
│ 22           │ TCP      │ Your IP     │ SSH access                          │
│ 6379         │ TCP      │ VPC only    │ Redis (internal only!)              │
│ 5000         │ TCP      │ Your IP     │ MLflow UI                           │
│ 8000         │ TCP      │ ALB SG      │ FastAPI (from load balancer)        │
│ 9090         │ TCP      │ Your IP     │ Prometheus                          │
│ 3000         │ TCP      │ Your IP     │ Grafana                             │
│ 443          │ TCP      │ 0.0.0.0/0   │ ALB HTTPS (public)                  │
└──────────────┴──────────┴─────────────┴─────────────────────────────────────┘

⚠️  IMPORTANT: Redis (6379) should NEVER be exposed to the internet!
    Keep it VPC-internal or use Redis AUTH password.
```
