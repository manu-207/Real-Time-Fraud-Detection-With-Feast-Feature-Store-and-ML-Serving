"""
monitor.py
===========
Runs Evidently AI drift detection on the feature data.
Compares a reference split (training data) against a current split
to detect feature drift and prediction drift.

Outputs:
- reports/drift_report.html — full Evidently HTML report
- reports/drift_summary.json — machine-readable drift metrics

Usage:
    python src/monitor.py
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()

# Evidently imports
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)

params = yaml.safe_load(open("params.yaml"))
data_params = params["data"]
train_params = params["train"]
monitor_params = params["monitoring"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))


def get_feature_columns():
    """Return the list of feature columns used for training."""
    v_cols = [f"v{i}" for i in range(1, 29)]
    engineered = ["scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"]
    return v_cols + engineered


def run_monitoring():
    """Run drift detection and generate reports."""
    processed_path = data_params["processed_path"]
    model_path = train_params["model_path"]
    report_dir = monitor_params["report_dir"]

    print(f"[monitor] Loading data from {processed_path}")
    df = pd.read_parquet(processed_path)

    feature_cols = get_feature_columns()

    # ── Load model and add predictions ────────────────────────────────────────
    print(f"[monitor] Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    X = df[feature_cols]
    df["prediction"] = model.predict(X)
    df["prediction_proba"] = model.predict_proba(X)[:, 1]

    # ── Split into reference (70%) and current (30%) ──────────────────────────
    ref_df, cur_df = train_test_split(df, test_size=0.3, random_state=42)
    print(f"[monitor] Reference: {len(ref_df):,} | Current: {len(cur_df):,}")

    # ── Build Evidently Report ────────────────────────────────────────────────
    # Select columns for drift analysis
    drift_columns = ["scaled_amount", "v1", "v3", "v14", "v17",
                     "hour_of_day", "is_night", "amount_zscore", "prediction_proba"]

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
        ColumnDriftMetric(column_name="scaled_amount"),
        ColumnDriftMetric(column_name="v14"),
        ColumnDriftMetric(column_name="prediction_proba"),
    ])

    report.run(
        reference_data=ref_df[drift_columns],
        current_data=cur_df[drift_columns],
    )

    # ── Save HTML report ──────────────────────────────────────────────────────
    os.makedirs(report_dir, exist_ok=True)
    html_path = os.path.join(report_dir, "drift_report.html")
    report.save_html(html_path)
    print(f"[monitor] HTML report saved to {html_path}")

    # ── Extract metrics ───────────────────────────────────────────────────────
    report_dict = report.as_dict()
    metrics = report_dict.get("metrics", [])

    drift_share = 0.0
    amount_drift = 0
    v14_drift = 0
    prediction_drift = 0

    for metric in metrics:
        metric_id = metric.get("metric", "")
        result = metric.get("result", {})

        if "DatasetDriftMetric" in str(metric_id):
            drift_share = result.get("share_of_drifted_columns", 0.0)
            dataset_drift = result.get("dataset_drift", False)

        elif "ColumnDriftMetric" in str(metric_id):
            col = result.get("column_name", "")
            drifted = result.get("drift_detected", False)
            if col == "scaled_amount":
                amount_drift = int(drifted)
            elif col == "v14":
                v14_drift = int(drifted)
            elif col == "prediction_proba":
                prediction_drift = int(drifted)

    summary = {
        "drift_share": round(drift_share, 4),
        "dataset_drift": int(dataset_drift) if 'dataset_drift' in dir() else 0,
        "amount_drift": amount_drift,
        "v14_drift": v14_drift,
        "prediction_drift": prediction_drift,
    }

    # ── Save JSON summary ─────────────────────────────────────────────────────
    json_path = os.path.join(report_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[monitor] JSON summary saved to {json_path}")
    print(f"[monitor] Drift metrics: {summary}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="drift-monitoring"):
        mlflow.log_metrics({
            "drift_share": summary["drift_share"],
            "amount_drift": summary["amount_drift"],
            "v14_drift": summary["v14_drift"],
            "prediction_drift": summary["prediction_drift"],
        })
        mlflow.log_artifact(html_path, artifact_path="evidently")
        mlflow.log_artifact(json_path, artifact_path="evidently")
        print("[monitor] Metrics + artifacts logged to MLflow")

    return summary


if __name__ == "__main__":
    run_monitoring()
