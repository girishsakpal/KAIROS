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
  Uses a SQLite backend (sqlite:///mlflow.db) by default, not the plain
  filesystem store — newer MLflow versions deprecated file-store-only
  tracking and raise at runtime if pointed at a bare directory like
  './mlruns'. That exception is caught too, so a tracking-backend problem
  never takes down a run whose model training already succeeded.
- STEP 5 CHANGES (Phase 3 findings): `plan_type` (1:1 relabel of contract_type) and
  `months_of_usage_history` (redundant with tenure, inflates SHAP) are dropped.
  Both models are now calibrated: the last 15% of train (chronologically) is held
  out as a validation slice, models fit on the first 85%, and an isotonic
  calibrator is fit on the validation slice. The test set never touches fitting
  or calibration. Metrics are reported before AND after calibration.
- SHAP explains the RAW (uncalibrated) XGBoost output. Isotonic calibration is a
  monotone transform of that score, so driver rankings are unaffected; only the
  probability shown to the user is recalibrated.
- Champion logic is registry-aware: if model_registry_row.json already holds a
  non-baseline champion (e.g. the LSTM) with a higher ROC-AUC, this script leaves
  it and churn_predictions_test.csv alone. XGBoost's explainable predictions are
  ALWAYS written to xgboost_predictions_test.csv so the dashboard can use SHAP
  drivers regardless of who the champion is.
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
                                          (champion only; untouched if a stronger non-baseline champion exists)
    xgboost_predictions_test.csv        — XGBoost calibrated probs + SHAP top_features (always written)
    xgboost_registry_row.json           — registry row for the XGBoost run (champion or challenger)
    calibration_curve.png               — reliability diagram, raw vs calibrated
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline, Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = ["region", "signup_channel", "gender",
                     "contract_type", "billing_cycle", "payment_method"]
ID_COLS = ["customer_id", "snapshot_date"]
LABEL_COL = "churn_label"

# Dropped per docs/Phase3_Findings.md section 2 (redundant features).
DROP_COLS = ["plan_type", "months_of_usage_history"]
VALIDATION_FRACTION = 0.15      # chronological slice from the END of train
BASELINE_ID_PREFIXES = ("xgb_churn", "logreg_churn")
MODEL_VERSION = "v2"            # v1 = pre-Step-5 feature set / uncalibrated


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_data(data_dir: str):
    """Returns fit/val/test splits. Validation = most recent 15% of train."""
    train = pd.read_csv(os.path.join(data_dir, "features_train.csv"),
                        parse_dates=["snapshot_date"])
    test = pd.read_csv(os.path.join(data_dir, "features_test.csv"),
                       parse_dates=["snapshot_date"])
    # Chronological order so the tail really is "most recent", never random.
    train = train.sort_values("snapshot_date", kind="mergesort").reset_index(drop=True)

    feature_cols = [c for c in train.columns if c not in ID_COLS + [LABEL_COL] + DROP_COLS]
    numeric_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    n_val = int(len(train) * VALIDATION_FRACTION)
    fit_df, val_df = train.iloc[:-n_val], train.iloc[-n_val:]
    if val_df[LABEL_COL].nunique() < 2:
        print("[warn] Validation slice has a single class — calibration will be unreliable.")

    return {
        "X_fit": fit_df[feature_cols], "y_fit": fit_df[LABEL_COL],
        "X_val": val_df[feature_cols], "y_val": val_df[LABEL_COL],
        "X_test": test[feature_cols], "y_test": test[LABEL_COL],
        "test_df": test, "numeric_cols": numeric_cols, "feature_cols": feature_cols,
    }


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
# Calibration
# ---------------------------------------------------------------------

def fit_calibrator(val_proba, y_val):
    """Isotonic regression mapping raw score -> calibrated probability.
    Fit ONLY on the validation slice (never train-fit rows, never test)."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(val_proba, y_val)
    return iso


def calibrate_and_report(name, pipe, data):
    """Returns (calibrator, raw_test_proba, cal_test_proba, raw_metrics, cal_metrics)."""
    val_raw = pipe.predict_proba(data["X_val"])[:, 1]
    test_raw = pipe.predict_proba(data["X_test"])[:, 1]
    iso = fit_calibrator(val_raw, data["y_val"])
    test_cal = iso.predict(test_raw)
    raw_m = evaluate(data["y_test"], test_raw)
    cal_m = evaluate(data["y_test"], test_cal)
    actual = float(np.mean(data["y_test"]))
    print(f"[{name}] mean predicted churn on test: raw {test_raw.mean():.3f} -> "
          f"calibrated {test_cal.mean():.3f} (actual {actual:.3f})")
    print(f"[{name}] raw        : {raw_m}")
    print(f"[{name}] calibrated : {cal_m}")
    return iso, test_raw, test_cal, raw_m, cal_m


def save_calibration_plot(out_dir, y_test, curves):
    """curves: {label: proba}. Reliability diagram; skipped if matplotlib missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        for lab, proba in curves.items():
            frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", label=lab)
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed churn rate")
        plt.title("Kairos — Reliability curve (test set)")
        plt.legend(fontsize=8)
        plt.tight_layout()
        path = os.path.join(out_dir, "calibration_curve.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[write] {path}")
    except Exception as e:
        print(f"[warn] Could not save calibration plot: {e}")


# ---------------------------------------------------------------------
# Logistic Regression baseline
# ---------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, numeric_cols):
    print("\n[LogisticRegression] Training baseline (on fit slice)...")
    pipe = Pipeline([
        ("preprocess", build_preprocessor(numeric_cols)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------
# XGBoost + SHAP (guarded — may not be installed)
# ---------------------------------------------------------------------

def train_xgboost(X_train, y_train, numeric_cols):
    try:
        import xgboost as xgb
    except ImportError:
        print("\n[XGBoost] 'xgboost' is not installed in this environment — "
              "skipping XGBoost + SHAP. Install with `pip install xgboost shap` "
              "and re-run to get the full comparison.")
        return None, None

    print("\n[XGBoost] Training (on fit slice)...")
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
    feature_names = pipe.named_steps["preprocess"].get_feature_names_out()
    return pipe, feature_names


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

def log_to_mlflow(model_name, params, metrics, tracking_uri="sqlite:///mlflow.db"):
    try:
        import mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("kairos_churn_baseline")
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
        print(f"[MLflow] Logged run for {model_name} to {tracking_uri}")
    except ImportError:
        print(f"\n[MLflow] 'mlflow' is not installed — skipping tracking for {model_name}. "
              "Install with `pip install mlflow` and re-run to get experiment tracking.")
    except Exception as e:
        # Broadened beyond ImportError on purpose: newer MLflow versions raise a
        # runtime MlflowException if pointed at a plain filesystem URI (the old
        # './mlruns' default is now deprecated in favor of a DB backend like
        # sqlite:///mlflow.db). Rather than let a tracking-backend issue take down
        # a run whose actual model training already succeeded, log a warning and
        # continue — metrics.json still has everything either way.
        print(f"\n[MLflow] Logging failed for {model_name}, continuing without it. "
              f"Reason: {e}")


# ---------------------------------------------------------------------
# DB-ready output files
# ---------------------------------------------------------------------

def write_model_registry_row(out_dir, model_id, algorithm, metrics, feature_names,
                              status="champion", filename="model_registry_row.json"):
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
        "status": status,
    }
    path = os.path.join(out_dir, filename)
    with open(path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"[write] {path}  (status={status})")


def write_churn_predictions(out_dir, test_df, proba, model_id, top_features_per_customer=None,
                             filename="churn_predictions_test.csv", proba_raw=None):
    rows = pd.DataFrame({
        "customer_id": test_df["customer_id"].values,
        "model_id": model_id,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "churn_probability": proba,
        "risk_tier": [risk_tier(p) for p in proba],
        "actual_churn": test_df[LABEL_COL].values,
    })
    if proba_raw is not None:
        rows["churn_probability_raw"] = proba_raw   # extra column, not part of DB table
    if top_features_per_customer is not None:
        rows["top_features"] = [json.dumps(tf) for tf in top_features_per_customer]
    else:
        rows["top_features"] = None

    path = os.path.join(out_dir, filename)
    rows.to_csv(path, index=False)
    print(f"[write] {path}  ({len(rows)} rows)")


def update_metrics_json(out_dir, new_entries):
    """Merge into metrics.json instead of overwriting, so LSTM entries written by
    train_lstm.py survive a re-run of this script."""
    path = os.path.join(out_dir, "metrics.json")
    existing = {}
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.update(new_entries)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n[write] {path} (merged; kept: {sorted(set(existing) - set(new_entries))})")
    return existing


def read_existing_registry(out_dir):
    path = os.path.join(out_dir, "model_registry_row.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    import joblib
    parser = argparse.ArgumentParser(description="Kairos baseline model training.")
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--mlflow-uri", type=str, default="sqlite:///mlflow.db")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_data(args.data_dir)
    print(f"[info] Fit:  {len(data['X_fit'])} rows, churn rate {data['y_fit'].mean():.1%}")
    print(f"[info] Val:  {len(data['X_val'])} rows, churn rate {data['y_val'].mean():.1%}  (calibration only)")
    print(f"[info] Test: {len(data['X_test'])} rows, churn rate {data['y_test'].mean():.1%}")
    print(f"[info] Dropped features: {DROP_COLS}")

    new_metrics, curves = {}, {}

    # --- Logistic Regression (always runs) ---
    lr_pipe = train_logistic_regression(data["X_fit"], data["y_fit"], data["numeric_cols"])
    lr_iso, lr_raw, lr_cal, lr_raw_m, lr_cal_m = calibrate_and_report("LogisticRegression", lr_pipe, data)
    new_metrics["logistic_regression"] = lr_raw_m
    new_metrics["logistic_regression_calibrated"] = lr_cal_m
    curves["LogReg raw"], curves["LogReg calibrated"] = lr_raw, lr_cal
    joblib.dump(lr_pipe, os.path.join(args.out_dir, "logistic_regression_pipeline.joblib"))
    joblib.dump(lr_iso, os.path.join(args.out_dir, "logistic_regression_calibrator.joblib"))
    log_to_mlflow("logistic_regression", {"class_weight": "balanced", "max_iter": 1000,
                                           "dropped": ",".join(DROP_COLS)},
                  {**lr_raw_m, **{f"cal_{k}": v for k, v in lr_cal_m.items()}}, args.mlflow_uri)

    # --- XGBoost (guarded) ---
    xgb_pipe, feature_names = train_xgboost(data["X_fit"], data["y_fit"], data["numeric_cols"])
    xgb_ready = xgb_pipe is not None
    top_features_per_customer = None
    if xgb_ready:
        xgb_iso, xgb_raw, xgb_cal, xgb_raw_m, xgb_cal_m = calibrate_and_report("XGBoost", xgb_pipe, data)
        new_metrics["xgboost"] = xgb_raw_m
        new_metrics["xgboost_calibrated"] = xgb_cal_m
        curves["XGBoost raw"], curves["XGBoost calibrated"] = xgb_raw, xgb_cal
        joblib.dump(xgb_pipe, os.path.join(args.out_dir, "xgboost_pipeline.joblib"))
        joblib.dump(xgb_iso, os.path.join(args.out_dir, "xgboost_calibrator.joblib"))
        log_to_mlflow("xgboost", {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05,
                                   "dropped": ",".join(DROP_COLS)},
                      {**xgb_raw_m, **{f"cal_{k}": v for k, v in xgb_cal_m.items()}}, args.mlflow_uri)
        _, top_features_per_customer = run_shap(xgb_pipe, data["X_test"], feature_names, args.out_dir)

    save_calibration_plot(args.out_dir, data["y_test"], curves)
    update_metrics_json(args.out_dir, new_metrics)

    # --- This run's best baseline, judged on the probabilities we would actually serve
    #     (calibrated). XGBoost wins ties, same as before.
    if xgb_ready and xgb_cal_m["roc_auc"] >= lr_cal_m["roc_auc"]:
        best = dict(id=f"xgb_churn_{MODEL_VERSION}", algo="XGBoost", metrics=xgb_cal_m,
                    proba=xgb_cal, raw=xgb_raw, feats=feature_names, tf=top_features_per_customer)
    else:
        best = dict(id=f"logreg_churn_{MODEL_VERSION}", algo="LogisticRegression", metrics=lr_cal_m,
                    proba=lr_cal, raw=lr_raw,
                    feats=lr_pipe.named_steps["preprocess"].get_feature_names_out(), tf=None)

    # --- Registry-aware champion check (fixes the "doesn't know the LSTM exists" trap) ---
    existing = read_existing_registry(args.out_dir)
    defer = False
    if existing:
        ex_id, ex_auc = str(existing.get("model_id", "")), existing.get("roc_auc")
        is_baseline = ex_id.startswith(BASELINE_ID_PREFIXES)
        print(f"\n[compare] Existing champion: {ex_id} (roc_auc={ex_auc}) "
              f"vs this run's best baseline: {best['id']} (calibrated roc_auc={best['metrics']['roc_auc']})")
        # Stale baseline rows (xgb_churn_v1 etc.) are always replaceable; only a
        # NON-baseline champion (the LSTM) can block promotion.
        if not is_baseline and ex_auc is not None and ex_auc >= best["metrics"]["roc_auc"]:
            defer = True

    if defer:
        print(f"[compare] {existing['model_id']} keeps the champion slot — "
              "model_registry_row.json and churn_predictions_test.csv left untouched.")
    else:
        write_model_registry_row(args.out_dir, best["id"], best["algo"], best["metrics"], best["feats"])
        write_churn_predictions(args.out_dir, data["test_df"], best["proba"], best["id"], best["tf"],
                                proba_raw=None)

    # --- XGBoost explainable outputs: ALWAYS written (dashboard + DB consume these) ---
    if xgb_ready:
        xgb_id = f"xgb_churn_{MODEL_VERSION}"
        status = "champion" if (not defer and best["id"] == xgb_id) else "challenger"
        write_model_registry_row(args.out_dir, xgb_id, "XGBoost", xgb_cal_m, feature_names,
                                  status=status, filename="xgboost_registry_row.json")
        write_churn_predictions(args.out_dir, data["test_df"], xgb_cal, xgb_id,
                                top_features_per_customer,
                                filename="xgboost_predictions_test.csv", proba_raw=xgb_raw)

    print(f"\n[done] Champion slot: {'kept ' + existing['model_id'] if defer else best['algo'] + ' (' + best['id'] + ')'}")
    if defer:
        print("[note] Dashboard data should be built from xgboost_predictions_test.csv "
              "(has SHAP drivers); the champion file has none.")


if __name__ == "__main__":
    main()
