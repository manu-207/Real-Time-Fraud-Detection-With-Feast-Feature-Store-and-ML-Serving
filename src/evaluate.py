"""
evaluate.py
============
Evaluates the trained XGBoost model on the full dataset and logs
metrics to MLflow. Generates detailed evaluation artifacts.

Usage:
    python src/evaluate.py
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()

params = yaml.safe_load(open("params.yaml"))
data_params = params["data"]
train_params = params["train"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection"))


def get_feature_columns():
    """Return the list of feature columns used for training."""
    v_cols = [f"v{i}" for i in range(1, 29)]
    engineered = ["scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"]
    return v_cols + engineered


def evaluate():
    """Evaluate model on full dataset and log to MLflow."""
    model_path = train_params["model_path"]
    processed_path = data_params["processed_path"]

    print(f"[evaluate] Loading model from {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    print(f"[evaluate] Loading data from {processed_path}")
    df = pd.read_parquet(processed_path)

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["is_fraud"]

    # ── Predictions ───────────────────────────────────────────────────────────
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # ── Metrics ───────────────────────────────────────────────────────────────
    roc_auc = roc_auc_score(y, y_prob)
    pr_auc = average_precision_score(y, y_prob)
    f1 = f1_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    cr = classification_report(y, y_pred, target_names=["Legit", "Fraud"])

    print(f"\n[evaluate] Full Dataset Evaluation:")
    print(f"  Samples: {len(X):,}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  F1:      {f1:.4f}")
    print(f"\n{cr}")

    # ── Threshold analysis ────────────────────────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    # Find threshold that maximizes F1
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    print(f"\n[evaluate] Optimal threshold: {best_threshold:.4f}")
    print(f"  Precision at optimal: {precision[best_idx]:.4f}")
    print(f"  Recall at optimal: {recall[best_idx]:.4f}")
    print(f"  F1 at optimal: {f1_scores[best_idx]:.4f}")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics({
            "eval_roc_auc": roc_auc,
            "eval_pr_auc": pr_auc,
            "eval_f1": f1,
            "eval_best_threshold": best_threshold,
            "eval_precision_at_optimal": float(precision[best_idx]),
            "eval_recall_at_optimal": float(recall[best_idx]),
        })
        mlflow.log_text(cr, "eval_classification_report.txt")

    # ── Save evaluation report ────────────────────────────────────────────────
    os.makedirs("reports", exist_ok=True)
    report = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1,
        "best_threshold": float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "total_samples": len(X),
        "fraud_samples": int(y.sum()),
    }
    with open("reports/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[evaluate] Report saved to reports/evaluation_report.json")


if __name__ == "__main__":
    evaluate()
