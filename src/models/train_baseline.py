"""
Kairos — Baseline Model Training (Phase 3)
=============================================

Trains the baseline Logistic Regression + XGBoost churn models on the
leakage-safe features from src/features/build_features.py, computes SHAP
explanations, logs everything to MLflow, and writes DB-ready output files
that map directly onto the `model_registry` and `churn_predictions` tables
in db/schema.sql.

Design notes
------------
- Logistic Regression is the interpretable baseline (Objective 2). It's
  trained and evaluated unconditionally.
- XGBoost is the main model. Import is guarded: if xgboost isn't installed,
  the script prints a clear message and skips XGBoost + SHAP rather than
  crashing, so the LR baseline still completes on a bare environment.
- SHAP (TreeExplainer) runs against the trained XGBoost model. Guarded the
  same way.
- MLflow logging is guarded the same way — if not installed, metrics/params
  are still printed and written to metrics.json, just not tracked in MLflow.
- class imbalance (churn ~22-42% depending on split) is handled via
  class_weight="balanced" (LR) and scale_pos_weight (XGBoost), not by
  resampling — keeps the leakage-safe row structure untouched.

Outputs (all under --out-dir, default `models/`):
    metrics.json                        — ROC-AUC, PR-AUC, Brier score per model
    logistic_regression_pipeline.joblib — fitted sklearn Pipeline
    xgboost_pipeline.joblib             — fitted sklearn Pipeline (if xgboost available)
    shap_global_importance.csv          — mean |SHAP value| per feature (if shap available)
    shap_summary_plot.png               — bar chart of global importance (if shap available)
    model_registry_row.json             — ready to insert into model_registry table
    churn_predictions_test.csv          — ready to insert into churn_predictions table
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline, Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = ["region", "signup_channel", "gender", "plan_type",
                     "contract_type", "billing_cycle", "payment_method"]
ID_COLS = ["customer_id", "snapshot_date"]
LABEL_COL = "churn_label"


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(data_dir: str):
    train = pd.read_csv(os.path.join(data_dir, "features_train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "features_test.csv"))
    feature_cols = [c for c in train.columns if c not in ID_COLS + [LABEL_COL]]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    X_train, y_train = train[feature_cols], train[LABEL_COL]
    X_test, y_test = test[feature_cols], test[LABEL_COL]
    return X_train, y_train, X_test, y_test, test, numeric_cols


def build_preprocessor(numeric_cols):
    numeric_pipeline = SkPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ("num", numeric_pipeline, numeric_cols),
    ])


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate(y_true, y_proba) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),  # lower = better calibrated
    }


def risk_tier(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


# ---------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, X_test, y_test, numeric_cols):
    print("\n[LogisticRegression] Training baseline...")
    pipe = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    print(f"[LogisticRegression] Test metrics: {metrics}")
    return pipe, proba, metrics


# ---------------------------------------------------------------------
# XGBoost + SHAP (guarded — may not be installed)
# ---------------------------------------------------------------------

def train_xgboost(X_train, y_train, X_test, y_test, numeric_cols):
    try:
        import xgboost as xgb
    except ImportError:
        print("\n[XGBoost] 'xgboost' is not installed in this environment — "
              "skipping XGBoost + SHAP. Install with `pip install xgboost shap` "
              "and re-run to get the full comparison.")
        return None, None, None, None

    print("\n[XGBoost] Training...")
    preprocessor = build_preprocessor(numeric_cols)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    pipe = Pipeline([
        ("preprocess", preprocessor),
        ("clf", xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    print(f"[XGBoost] Test metrics: {metrics}")

    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    return pipe, proba, metrics, feature_names


def run_shap(pipe, X_test, feature_names, out_dir, top_k: int = 3):
    try:
        import shap
    except ImportError:
        print("\n[SHAP] 'shap' is not installed in this environment — skipping "
              "explanation generation. Install with `pip install shap` and re-run.")
        return None, None

    print("\n[SHAP] Computing TreeExplainer values on the test set...")
    X_test_transformed = pipe.named_steps["preprocess"].transform(X_test)
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    explainer = shap.TreeExplainer(pipe.named_steps["clf"])
    shap_values = explainer.shap_values(X_test_transformed)

    # Global importance
    global_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False)

    global_path = os.path.join(out_dir, "shap_global_importance.csv")
    global_importance.to_csv(global_path, index=False)
    print(f"[write] {global_path}")

    try:
        import matplotlib.pyplot as plt
        top20 = global_importance.head(20).iloc[::-1]
        plt.figure(figsize=(8, 6))
        plt.barh(top20["feature"], top20["mean_abs_shap"])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Kairos — Global Feature Importance (XGBoost + SHAP)")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "shap_summary_plot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[write] {plot_path}")
    except Exception as e:
        print(f"[warn] Could not save SHAP plot: {e}")

    # Per-customer top-k drivers, in the exact shape churn_predictions.top_features expects
    per_customer_top_features = []
    for i in range(shap_values.shape[0]):
        row_shap = shap_values[i]
        top_idx = np.argsort(np.abs(row_shap))[::-1][:top_k]
        drivers = [
            {
                "feature": str(feature_names[j]),
                "shap_value": round(float(row_shap[j]), 4),
                "direction": "increases_risk" if row_shap[j] > 0 else "decreases_risk",
            }
            for j in top_idx
        ]
        per_customer_top_features.append(drivers)

    return global_importance, per_customer_top_features


# ---------------------------------------------------------------------
# MLflow logging (guarded — may not be installed)
# ---------------------------------------------------------------------

def log_to_mlflow(model_name, params, metrics, tracking_uri="./mlruns"):
    try:
        import mlflow
    except ImportError:
        print(f"\n[MLflow] 'mlflow' is not installed — skipping tracking for {model_name}. "
              "Install with `pip install mlflow` and re-run to get experiment tracking.")
        return

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("kairos_churn_baseline")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
    print(f"[MLflow] Logged run for {model_name} to {tracking_uri}")


# ---------------------------------------------------------------------
# DB-ready output files
# ---------------------------------------------------------------------

def write_model_registry_row(out_dir, model_id, algorithm, metrics, feature_names):
    row = {
        "model_id": model_id,
        "model_name": f"Kairos churn model ({algorithm})",
        "algorithm": algorithm,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_list": list(feature_names),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "calibration_error": metrics["brier_score"],
        "drift_psi_score": None,   # populated by Phase 6 drift monitoring, not at training time
        "drift_ks_pvalue": None,
        "drift_detected": False,
        "status": "champion",
    }
    path = os.path.join(out_dir, "model_registry_row.json")
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"[write] {path}")


def write_churn_predictions(out_dir, test_df, proba, model_id, top_features_per_customer=None):
    rows = pd.DataFrame({
        "customer_id": test_df["customer_id"].values,
        "model_id": model_id,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "churn_probability": proba,
        "risk_tier": [risk_tier(p) for p in proba],
        "actual_churn": test_df[LABEL_COL].values,
    })
    if top_features_per_customer is not None:
        rows["top_features"] = [json.dumps(tf) for tf in top_features_per_customer]
    else:
        rows["top_features"] = None

    path = os.path.join(out_dir, "churn_predictions_test.csv")
    rows.to_csv(path, index=False)
    print(f"[write] {path}  ({len(rows)} rows — ready for \\copy into churn_predictions, "
          f"after model_registry_row.json is inserted first for the FK)")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kairos baseline model training.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--mlflow-uri", type=str, default="./mlruns")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    X_train, y_train, X_test, y_test, test_df, numeric_cols = load_data(args.data_dir)
    print(f"[info] Train: {len(X_train)} rows, churn rate {y_train.mean():.1%}")
    print(f"[info] Test:  {len(X_test)} rows, churn rate {y_test.mean():.1%}")

    all_metrics = {}

    # --- Logistic Regression (always runs) ---
    import joblib
    lr_pipe, lr_proba, lr_metrics = train_logistic_regression(
        X_train, y_train, X_test, y_test, numeric_cols)
    all_metrics["logistic_regression"] = lr_metrics
    joblib.dump(lr_pipe, os.path.join(args.out_dir, "logistic_regression_pipeline.joblib"))
    log_to_mlflow("logistic_regression",
                   {"class_weight": "balanced", "max_iter": 1000},
                   lr_metrics, args.mlflow_uri)

    # --- XGBoost (guarded) ---
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test, numeric_cols)
    xgb_pipe, xgb_proba, xgb_metrics, feature_names = xgb_result if xgb_result[0] else (None, None, None, None)

    top_features_per_customer = None
    if xgb_pipe is not None:
        all_metrics["xgboost"] = xgb_metrics
        joblib.dump(xgb_pipe, os.path.join(args.out_dir, "xgboost_pipeline.joblib"))
        log_to_mlflow("xgboost",
                       {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
                       xgb_metrics, args.mlflow_uri)

        global_importance, top_features_per_customer = run_shap(
            xgb_pipe, X_test, feature_names, args.out_dir)

    # --- Metrics summary ---
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[write] {metrics_path}")
    print(f"[summary] {json.dumps(all_metrics, indent=2)}")

    # --- Champion model choice: XGBoost if available and it beats LR on ROC-AUC, else LR ---
    if xgb_metrics and xgb_metrics["roc_auc"] >= lr_metrics["roc_auc"]:
        champion_id = "xgb_churn_v1"
        champion_algo = "XGBoost"
        champion_metrics = xgb_metrics
        champion_proba = xgb_proba
        champion_features = feature_names
    else:
        champion_id = "logreg_churn_v1"
        champion_algo = "LogisticRegression"
        champion_metrics = lr_metrics
        champion_proba = lr_proba
        champion_features = lr_pipe.named_steps["preprocess"].get_feature_names_out()

    write_model_registry_row(args.out_dir, champion_id, champion_algo,
                              champion_metrics, champion_features)
    write_churn_predictions(args.out_dir, test_df, champion_proba, champion_id,
                             top_features_per_customer if champion_algo == "XGBoost" else None)

    print(f"\n[done] Champion model: {champion_algo} ({champion_id})")


if __name__ == "__main__":
    main()
