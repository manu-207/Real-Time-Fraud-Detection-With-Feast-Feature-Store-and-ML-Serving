"""
train.py
=========
Trains an XGBoost classifier for fraud detection.
- Handles extreme class imbalance via scale_pos_weight
- Uses early stopping on validation set
- Logs all params, metrics, and model to MLflow
- Saves model locally as model.json (XGBoost native format)

Usage:
    python src/train.py
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
)
import mlflow
import mlflow.xgboost
import yaml
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
params = yaml.safe_load(open("params.yaml"))
data_params = params["data"]
train_params = params["train"]
xgb_params = train_params["xgboost"]

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud-detection")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


def get_feature_columns():
    """Return the list of feature columns used for training."""
    v_cols = [f"v{i}" for i in range(1, 29)]
    engineered = ["scaled_amount", "scaled_time", "hour_of_day", "is_night", "amount_zscore"]
    return v_cols + engineered


def train():
    """Train XGBoost model with MLflow tracking."""
    processed_path = data_params["processed_path"]
    model_path = train_params["model_path"]
    random_state = train_params["random_state"]

    print(f"[train] Loading features from {processed_path}")
    df = pd.read_parquet(processed_path)

    feature_cols = get_feature_columns()
    X = df[feature_cols]
    y = df["is_fraud"]

    print(f"[train] Features: {len(feature_cols)}, Samples: {len(X):,}")
    print(f"[train] Fraud ratio: {y.mean():.4%}")

    # ── Train/test split ──────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_params["test_size"],
        random_state=random_state, stratify=y
    )

    print(f"[train] Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"[train] Train fraud: {y_train.sum():,} | Test fraud: {y_test.sum():,}")

    # ── MLflow run ────────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="xgboost-fraud-detection"):
        # Log parameters
        mlflow.log_params({
            "n_estimators": xgb_params["n_estimators"],
            "max_depth": xgb_params["max_depth"],
            "learning_rate": xgb_params["learning_rate"],
            "scale_pos_weight": xgb_params["scale_pos_weight"],
            "subsample": xgb_params["subsample"],
            "colsample_bytree": xgb_params["colsample_bytree"],
            "eval_metric": xgb_params["eval_metric"],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "n_features": len(feature_cols),
            "fraud_ratio": f"{y.mean():.4%}",
        })

        # ── Train XGBoost ─────────────────────────────────────────────────────
        model = xgb.XGBClassifier(
            n_estimators=xgb_params["n_estimators"],
            max_depth=xgb_params["max_depth"],
            learning_rate=xgb_params["learning_rate"],
            scale_pos_weight=xgb_params["scale_pos_weight"],
            subsample=xgb_params["subsample"],
            colsample_bytree=xgb_params["colsample_bytree"],
            eval_metric=xgb_params["eval_metric"],
            early_stopping_rounds=xgb_params["early_stopping_rounds"],
            random_state=random_state,
            use_label_encoder=False,
            tree_method="hist",  # fast histogram-based training
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=20,
        )

        # ── Evaluate ──────────────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"])

        print(f"\n[train] Results:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  F1:      {f1:.4f}")
        print(f"\n{cr}")
        print(f"\nConfusion Matrix:\n{cm}")

        # Log metrics
        mlflow.log_metrics({
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "true_positives": int(cm[1, 1]),
            "false_positives": int(cm[0, 1]),
            "true_negatives": int(cm[0, 0]),
            "false_negatives": int(cm[1, 0]),
            "best_iteration": model.best_iteration,
        })

        # Log artifacts
        mlflow.log_text(cr, "classification_report.txt")
        mlflow.log_text(str(cm), "confusion_matrix.txt")

        # Feature importance
        importance = model.feature_importances_
        importance_dict = dict(zip(feature_cols, importance.tolist()))
        importance_sorted = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        mlflow.log_text(json.dumps(importance_sorted, indent=2), "feature_importance.json")

        # Log top 10 features
        print("\n[train] Top 10 features:")
        for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10]):
            print(f"  {i+1}. {feat}: {imp:.4f}")

        # ── Log model to MLflow ───────────────────────────────────────────────
        mlflow.xgboost.log_model(
            model, "model",
            registered_model_name="fraud-detection-xgboost",
        )

        # ── Save model locally ────────────────────────────────────────────────
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        print(f"\n[train] Model saved to {model_path}")

        # Save feature columns list for serving
        meta = {
            "feature_columns": feature_cols,
            "model_path": model_path,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "best_iteration": model.best_iteration,
        }
        meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[train] Model metadata saved to {meta_path}")

    return model


if __name__ == "__main__":
    train()
