"""
Kairos — LSTM Churn Model (Phase 3, third model alongside LR + XGBoost)
========================================================================

CAVEAT UP FRONT: TensorFlow is not installed in the sandbox this was written
in (no network access), so unlike train_baseline.py's LR path, nothing here
has actually been executed. It is written correctly against the Keras
Functional API and follows the same data contracts as build_features.py /
train_baseline.py, but please run it yourself and report back if anything
breaks — the same caveat that applied to xgboost/shap/mlflow in this repo.

WHY AN LSTM HERE, AND WHY HYBRID (not sequence-only)
-----------------------------------------------------
Per feature_manifest.json, the customer-level data in this project is
cross-sectional (one snapshot per customer) — that's what LR/XGBoost use,
and there's no sequence for a plain LSTM to exploit there. The one place a
real sequence exists is usage_logs.csv: up to 6 monthly rows per customer
(session_count, avg_session_minutes, feature_adoption_score).

An LSTM trained on ONLY that 6-step sequence would ignore contract_type,
tenure, and payment_method — the strongest signals in this dataset per
shap_global_importance.csv — and would likely lose to LR/XGBoost trivially,
which wouldn't be a meaningful comparison. So this script builds a HYBRID
model:
    - an LSTM branch reads the raw 6-month usage sequence
    - a dense branch reads the same static/tabular features LR & XGBoost use
      (MINUS the usage-aggregate columns build_features.py derived from that
      same sequence — avg_session_count_recent, session_count_trend, etc. —
      since the LSTM branch now learns that signal directly instead)
    - the two branches are concatenated before the final sigmoid

This makes the comparison fair and specific: "does learning the usage
trajectory directly beat summarizing it into 6 hand-built features?" rather
than "does a model with less information do worse" (which would be true but
uninteresting).

Per docs/Phase3_Findings.md, plan_type is a 1:1 relabel of contract_type
(zero exceptions in the data) and months_of_usage_history is redundant with
tenure_months — both are dropped from the static branch here. If you later
patch train_baseline.py to drop them too, this script needs no change; it
already does.

LEAKAGE / SPLIT — reused, not recomputed
------------------------------------------
This script does NOT recompute the snapshot-cutoff or train/test split
logic — it reads customer_id, snapshot_date, and split membership straight
from data/processed/features_{train,test}.csv (build_features.py's output),
so every model in this repo is compared on the *exact* same customers and
the exact same time-based split. It separately reads data/synthetic/
usage_logs.csv only to reconstruct each customer's usage sequence up to
their own snapshot_date (same "nothing after cutoff" rule build_features.py
already enforces for the aggregate features).

CALIBRATION
------------
Per docs/Phase3_Findings.md §3, LR and XGBoost's class-weighting inflates
predicted probabilities (mean predicted 0.565 vs actual 0.409 on test).
The same effect will happen here since this script also uses
class_weight="balanced". To not repeat that mistake a third time, this
script fits an isotonic calibrator on the held-out chronological validation
slice (carved from the END of train, never seen during weight updates) and
reports metrics BOTH before and after calibration. If you patch
train_baseline.py for calibration later, use the same approach (fit on a
validation slice, never on the test set).

Outputs (under --out-dir, default `models/`):
    lstm_model.keras                 — full Keras model (architecture+weights)
    lstm_static_preprocessor.joblib  — fitted ColumnTransformer (static branch)
    lstm_sequence_scaler.joblib      — fitted StandardScaler (sequence channels)
    lstm_calibrator.joblib           — fitted IsotonicRegression (may be None)
    lstm_permutation_importance.csv  — global importance (no SHAP for Keras
                                        here — see run_permutation_importance())
    lstm_predictions_test.csv        — same shape as churn_predictions_test.csv
    metrics.json                     — updated in place: adds an "lstm" key,
                                        preserves whatever train_baseline.py
                                        already wrote (LR / XGBoost)
    model_registry_row.json          — OVERWRITTEN ONLY if LSTM's test ROC-AUC
                                        beats the model currently recorded
                                        there (the reigning champion). Prints
                                        the comparison either way; never
                                        silently promotes or silently skips.
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
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ---------------------------------------------------------------------
# Config — must stay consistent with build_features.py's output columns
# ---------------------------------------------------------------------

MAX_SEQ_LEN = 6
SEQ_CHANNELS = ["session_count", "avg_session_minutes", "feature_adoption_score"]
PAD_SENTINEL_RAW = -999.0     # placeholder before scaling (outside any real range)
PAD_SENTINEL_SCALED = -10.0   # placeholder after scaling; Masking layer keys on this

# Usage-aggregate columns build_features.py derived FROM usage_logs — dropped
# from the static branch because the LSTM branch now learns that signal
# directly from the raw sequence instead. Keeping both would double-count it.
USAGE_AGGREGATE_COLS = [
    "avg_session_count_recent", "avg_session_minutes_recent",
    "avg_feature_adoption_recent", "session_count_trend",
    "days_since_last_active", "months_of_usage_history",
]
# Per docs/Phase3_Findings.md — plan_type is a 1:1 relabel of contract_type.
REDUNDANT_STATIC_COLS = ["plan_type"]

STATIC_CATEGORICAL_COLS = ["region", "signup_channel", "gender",
                            "contract_type", "billing_cycle", "payment_method"]
ID_COLS = ["customer_id", "snapshot_date"]
LABEL_COL = "churn_label"

VALIDATION_FRACTION = 0.15  # chronological slice carved from the end of train


# ---------------------------------------------------------------------
# 1. Load features_train/test.csv (already leakage-safe + split)
# ---------------------------------------------------------------------

def load_static_data(features_dir: str):
    train = pd.read_csv(os.path.join(features_dir, "features_train.csv"),
                         parse_dates=["snapshot_date"])
    test = pd.read_csv(os.path.join(features_dir, "features_test.csv"),
                        parse_dates=["snapshot_date"])
    # Chronological order matters: Keras's validation_split takes the TAIL of
    # the array it's given, so train must already be sorted by snapshot_date
    # for that tail to be a genuine "most recent 15% of train" slice, not a
    # random one.
    train = train.sort_values("snapshot_date").reset_index(drop=True)

    drop_cols = USAGE_AGGREGATE_COLS + REDUNDANT_STATIC_COLS
    static_cols = [c for c in train.columns
                   if c not in ID_COLS + [LABEL_COL] + drop_cols]
    numeric_cols = [c for c in static_cols if c not in STATIC_CATEGORICAL_COLS]

    return train, test, static_cols, numeric_cols


def build_static_preprocessor(numeric_cols):
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), STATIC_CATEGORICAL_COLS),
        ("num", numeric_pipeline, numeric_cols),
    ])


# ---------------------------------------------------------------------
# 2. Build the usage-log sequence tensor, respecting each customer's
#    own snapshot_date cutoff (same leakage rule as build_features.py)
# ---------------------------------------------------------------------

def load_usage_logs(raw_dir: str) -> pd.DataFrame:
    usage = pd.read_csv(os.path.join(raw_dir, "usage_logs.csv"),
                         parse_dates=["log_month", "last_active_date"])
    return usage


def build_sequences(customers_df: pd.DataFrame, usage_logs: pd.DataFrame) -> np.ndarray:
    """
    customers_df: must have customer_id, snapshot_date columns, one row per
    customer, in the order the caller wants the output array rows to match.

    Returns an array of shape (n_customers, MAX_SEQ_LEN, len(SEQ_CHANNELS)),
    raw (unscaled) values, right-padded with PAD_SENTINEL_RAW for customers
    with fewer than MAX_SEQ_LEN valid months before their snapshot_date.
    """
    merged = usage_logs.merge(
        customers_df[["customer_id", "snapshot_date"]], on="customer_id", how="inner"
    )
    valid = merged[merged["log_month"] <= merged["snapshot_date"]].copy()
    valid = valid.sort_values(["customer_id", "log_month"])

    by_customer = {cid: g[SEQ_CHANNELS].to_numpy(dtype=float)
                   for cid, g in valid.groupby("customer_id")}

    n = len(customers_df)
    out = np.full((n, MAX_SEQ_LEN, len(SEQ_CHANNELS)), PAD_SENTINEL_RAW, dtype=float)
    n_with_no_history = 0
    for i, cid in enumerate(customers_df["customer_id"].values):
        seq = by_customer.get(cid)
        if seq is None or len(seq) == 0:
            n_with_no_history += 1
            continue
        seq = seq[-MAX_SEQ_LEN:]          # keep the most recent up to 6 valid months
        out[i, : len(seq), :] = seq       # right-pad the rest with the sentinel
    if n_with_no_history:
        print(f"[warn] {n_with_no_history} customers have zero valid usage-log rows "
              f"before their snapshot_date — their sequence is fully padded "
              f"(the LSTM branch sees nothing for them; the static branch still does).")
    return out


def fit_sequence_scaler(seq_train_raw: np.ndarray) -> StandardScaler:
    """Fit on real (non-padded) timesteps only, so padding never pollutes the mean/std."""
    flat = seq_train_raw.reshape(-1, len(SEQ_CHANNELS))
    real_mask = ~np.all(flat == PAD_SENTINEL_RAW, axis=1)
    scaler = StandardScaler().fit(flat[real_mask])
    return scaler


def apply_sequence_scaler(seq_raw: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    """Scale real timesteps, then explicitly reset padded ones to a fixed
    sentinel the Masking layer keys on (scaling would otherwise map the raw
    -999 sentinel to some arbitrary, non-constant scaled value)."""
    shape = seq_raw.shape
    flat = seq_raw.reshape(-1, len(SEQ_CHANNELS))
    real_mask = ~np.all(flat == PAD_SENTINEL_RAW, axis=1)
    out = np.full_like(flat, PAD_SENTINEL_SCALED)
    out[real_mask] = scaler.transform(flat[real_mask])
    return out.reshape(shape)


# ---------------------------------------------------------------------
# 3. Model
# ---------------------------------------------------------------------

def build_model(n_static_features: int, lstm_units: int = 32):
    from tensorflow import keras
    from tensorflow.keras import layers

    seq_input = keras.Input(shape=(MAX_SEQ_LEN, len(SEQ_CHANNELS)), name="usage_sequence")
    x = layers.Masking(mask_value=PAD_SENTINEL_SCALED)(seq_input)
    x = layers.LSTM(lstm_units)(x)
    x = layers.Dropout(0.3)(x)

    static_input = keras.Input(shape=(n_static_features,), name="static_features")
    y = layers.Dense(32, activation="relu")(static_input)
    y = layers.Dropout(0.3)(y)

    combined = layers.Concatenate()([x, y])
    z = layers.Dense(16, activation="relu")(combined)
    z = layers.Dropout(0.2)(z)
    output = layers.Dense(1, activation="sigmoid")(z)

    model = keras.Model(inputs=[seq_input, static_input], outputs=output, name="kairos_lstm_hybrid")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc")],
    )
    return model


# ---------------------------------------------------------------------
# 4. Metrics + calibration (mirrors train_baseline.py's evaluate/risk_tier)
# ---------------------------------------------------------------------

def evaluate(y_true, y_proba) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
        "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
    }


def risk_tier(prob: float) -> str:
    if prob >= 0.6:
        return "high"
    if prob >= 0.3:
        return "medium"
    return "low"


def fit_calibrator(val_proba: np.ndarray, y_val: np.ndarray):
    """Isotonic calibration fit on the held-out chronological validation
    slice only — never on test, and never on data the model's weights were
    updated against. Returns None if the validation slice has only one
    class (can happen on small/degenerate splits) — isotonic needs both."""
    if len(np.unique(y_val)) < 2:
        print("[warn] Validation slice has only one class present — skipping "
              "calibration for this run. Predicted probabilities will keep "
              "the balanced-class-weight inflation described in "
              "docs/Phase3_Findings.md §3.")
        return None
    return IsotonicRegression(out_of_bounds="clip").fit(val_proba, y_val)


# ---------------------------------------------------------------------
# 5. Permutation importance (no SHAP DeepExplainer setup here — this is
#    a simpler, model-agnostic stand-in, comparable in spirit but NOT the
#    same thing as shap_global_importance.csv; label it clearly as such)
# ---------------------------------------------------------------------

def run_permutation_importance(model, seq_test, static_test_raw, y_test, static_cols,
                                n_repeats: int = 3, seed: int = 42) -> pd.DataFrame:
    """static_test_raw must be the RAW (pre-preprocessing) static dataframe —
    e.g. test[static_cols] — since per-column shuffling has to happen on the
    original columns (contract_type, tenure_months, ...), not on the one-hot/
    scaled matrix. It's run through the cached preprocessor internally,
    including for the baseline score, so this never feeds raw strings
    straight into the model."""
    rng = np.random.default_rng(seed)
    preprocessor = _static_preprocessor_transform_cache[0]
    X_static_test = preprocessor.transform(static_test_raw)
    if hasattr(X_static_test, "toarray"):
        X_static_test = X_static_test.toarray()

    baseline_auc = roc_auc_score(
        y_test, model.predict({"usage_sequence": seq_test, "static_features": X_static_test},
                               verbose=0).ravel()
    )
    rows = []

    # Whole-sequence permutation: shuffle entire per-customer sequences among
    # test rows, to see how much the LSTM branch as a whole contributes.
    drops = []
    for _ in range(n_repeats):
        perm = rng.permutation(len(seq_test))
        proba = model.predict(
            {"usage_sequence": seq_test[perm], "static_features": X_static_test}, verbose=0
        ).ravel()
        drops.append(baseline_auc - roc_auc_score(y_test, proba))
    rows.append({"feature": "usage_sequence_6mo (whole branch)",
                  "auc_drop_when_shuffled": round(float(np.mean(drops)), 4)})

    # Per-column permutation for the RAW static columns (before one-hot
    # encoding), so importance is reported per business-meaningful column
    # rather than per dummy variable.
    for col in static_cols:
        drops = []
        for _ in range(n_repeats):
            shuffled = static_test_raw.copy()
            shuffled[col] = rng.permutation(shuffled[col].values)
            X_shuffled = preprocessor.transform(shuffled)
            if hasattr(X_shuffled, "toarray"):
                X_shuffled = X_shuffled.toarray()
            proba = model.predict(
                {"usage_sequence": seq_test, "static_features": X_shuffled}, verbose=0
            ).ravel()
            drops.append(baseline_auc - roc_auc_score(y_test, proba))
        rows.append({"feature": col, "auc_drop_when_shuffled": round(float(np.mean(drops)), 4)})

    return pd.DataFrame(rows).sort_values("auc_drop_when_shuffled", ascending=False)


# module-level cache so run_permutation_importance can re-run the static
# preprocessor without threading it through every function signature
_static_preprocessor_transform_cache = [None]


# ---------------------------------------------------------------------
# 6. Champion comparison + DB-ready output files
# ---------------------------------------------------------------------

def maybe_promote_champion(out_dir, model_id, algorithm, metrics, feature_list,
                            test_df, proba):
    registry_path = os.path.join(out_dir, "model_registry_row.json")
    current_champion_auc = None
    if os.path.exists(registry_path):
        with open(registry_path) as f:
            current = json.load(f)
        current_champion_auc = current.get("roc_auc")
        print(f"[compare] Current champion: {current.get('algorithm')} "
              f"(roc_auc={current_champion_auc}) vs LSTM (roc_auc={metrics['roc_auc']})")

    if current_champion_auc is not None and metrics["roc_auc"] <= current_champion_auc:
        print("[compare] LSTM did not beat the current champion on test ROC-AUC — "
              "model_registry_row.json and churn_predictions_test.csv left untouched. "
              "LSTM's own outputs (lstm_model.keras, lstm_predictions_test.csv, etc.) "
              "are still written for the comparison table in your report.")
        return False

    print("[compare] LSTM beats the current champion (or none was recorded yet) — "
          "promoting: overwriting model_registry_row.json and churn_predictions_test.csv.")
    row = {
        "model_id": model_id,
        "model_name": f"Kairos churn model ({algorithm})",
        "algorithm": algorithm,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_list": list(feature_list),
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "calibration_error": metrics["brier_score"],
        "drift_psi_score": None,
        "drift_ks_pvalue": None,
        "drift_detected": False,
        "status": "champion",
    }
    with open(registry_path, "w") as f:
        json.dump(row, f, indent=2)
    print(f"[write] {registry_path}")

    preds = pd.DataFrame({
        "customer_id": test_df["customer_id"].values,
        "model_id": model_id,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "churn_probability": proba,
        "risk_tier": [risk_tier(p) for p in proba],
        "actual_churn": test_df[LABEL_COL].values,
        "top_features": None,  # no SHAP-equivalent per-customer explanation for this model yet
    })
    preds_path = os.path.join(out_dir, "churn_predictions_test.csv")
    preds.to_csv(preds_path, index=False)
    print(f"[write] {preds_path}")
    return True


def write_lstm_predictions(out_dir, test_df, proba_raw, proba_calibrated, model_id):
    preds = pd.DataFrame({
        "customer_id": test_df["customer_id"].values,
        "model_id": model_id,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
        "churn_probability_raw": proba_raw,
        "churn_probability_calibrated": proba_calibrated if proba_calibrated is not None else proba_raw,
        "risk_tier": [risk_tier(p) for p in (proba_calibrated if proba_calibrated is not None else proba_raw)],
        "actual_churn": test_df[LABEL_COL].values,
    })
    path = os.path.join(out_dir, "lstm_predictions_test.csv")
    preds.to_csv(path, index=False)
    print(f"[write] {path}")


def update_metrics_json(out_dir, metrics_raw, metrics_calibrated):
    path = os.path.join(out_dir, "metrics.json")
    all_metrics = {}
    if os.path.exists(path):
        with open(path) as f:
            all_metrics = json.load(f)
    all_metrics["lstm"] = metrics_raw
    all_metrics["lstm_calibrated"] = metrics_calibrated if metrics_calibrated else metrics_raw
    with open(path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[write] {path} (added 'lstm' / 'lstm_calibrated', preserved existing entries)")


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kairos hybrid LSTM churn model.")
    parser.add_argument("--features-dir", type=str, default="data/processed",
                         help="Where features_train.csv / features_test.csv live.")
    parser.add_argument("--raw-dir", type=str, default="data/synthetic",
                         help="Where usage_logs.csv lives (for the raw sequence).")
    parser.add_argument("--out-dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lstm-units", type=int, default=32)
    args = parser.parse_args()

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError:
        print("[LSTM] 'tensorflow' is not installed in this environment — skipping. "
              "Install with `pip install tensorflow` (CPU build is fine for this "
              "dataset size) and re-run.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    import joblib

    # --- 1. Load the SAME split LR/XGBoost use ---
    train, test, static_cols, numeric_cols = load_static_data(args.features_dir)
    print(f"[info] Train: {len(train)} rows, churn rate {train[LABEL_COL].mean():.1%}")
    print(f"[info] Test:  {len(test)} rows, churn rate {test[LABEL_COL].mean():.1%}")
    print(f"[info] Static features used (usage-aggregates + plan_type dropped, "
          f"see module docstring): {static_cols}")

    # --- 2. Static preprocessing (fit on train only) ---
    static_preprocessor = build_static_preprocessor(numeric_cols)
    X_static_train = static_preprocessor.fit_transform(train[static_cols])
    X_static_test = static_preprocessor.transform(test[static_cols])
    if hasattr(X_static_train, "toarray"):
        X_static_train = X_static_train.toarray()
        X_static_test = X_static_test.toarray()
    _static_preprocessor_transform_cache[0] = static_preprocessor

    # --- 3. Sequence build (leakage-safe: cutoff per customer's own snapshot_date) ---
    usage_logs = load_usage_logs(args.raw_dir)
    seq_train_raw = build_sequences(train, usage_logs)
    seq_test_raw = build_sequences(test, usage_logs)
    seq_scaler = fit_sequence_scaler(seq_train_raw)
    X_seq_train = apply_sequence_scaler(seq_train_raw, seq_scaler)
    X_seq_test = apply_sequence_scaler(seq_test_raw, seq_scaler)

    y_train = train[LABEL_COL].values.astype(float)
    y_test = test[LABEL_COL].values.astype(float)

    # --- 4. Chronological validation slice (tail of already-sorted train) ---
    n_val = int(len(train) * VALIDATION_FRACTION)
    val_slice = slice(len(train) - n_val, len(train))
    train_slice = slice(0, len(train) - n_val)

    class_weights = compute_class_weight("balanced", classes=np.array([0, 1]),
                                          y=y_train[train_slice])
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    print(f"[info] class_weight (balanced, fit on the training portion only): {class_weight_dict}")

    # --- 5. Build + train ---
    model = build_model(n_static_features=X_static_train.shape[1], lstm_units=args.lstm_units)
    model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", patience=5, restore_best_weights=True
    )
    history = model.fit(
        {"usage_sequence": X_seq_train, "static_features": X_static_train},
        y_train,
        validation_split=VALIDATION_FRACTION,  # takes the TAIL of the arrays above —
                                                # correct only because train was sorted
                                                # by snapshot_date in load_static_data()
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        shuffle=True,  # shuffles minibatch order only, applied AFTER the
                       # validation slice above is carved off; does not
                       # affect which rows ended up in train vs validation
        verbose=2,
    )

    # --- 6. Evaluate on test (raw, uncalibrated) ---
    proba_test_raw = model.predict(
        {"usage_sequence": X_seq_test, "static_features": X_static_test}, verbose=0
    ).ravel()
    metrics_raw = evaluate(y_test, proba_test_raw)
    print(f"[LSTM] Test metrics (raw, uncalibrated): {metrics_raw}")

    # --- 7. Calibrate on the held-out validation slice, re-evaluate on test ---
    proba_val_raw = model.predict(
        {"usage_sequence": X_seq_train[val_slice], "static_features": X_static_train[val_slice]},
        verbose=0,
    ).ravel()
    calibrator = fit_calibrator(proba_val_raw, y_train[val_slice])
    proba_test_calibrated = calibrator.transform(proba_test_raw) if calibrator is not None else None
    metrics_calibrated = evaluate(y_test, proba_test_calibrated) if calibrator is not None else None
    if metrics_calibrated:
        print(f"[LSTM] Test metrics (isotonic-calibrated): {metrics_calibrated}")
        print(f"[LSTM] Mean predicted before/after calibration: "
              f"{proba_test_raw.mean():.3f} -> {proba_test_calibrated.mean():.3f} "
              f"(actual test churn rate: {y_test.mean():.3f})")

    # --- 8. Save artifacts ---
    model.save(os.path.join(args.out_dir, "lstm_model.keras"))
    joblib.dump(static_preprocessor, os.path.join(args.out_dir, "lstm_static_preprocessor.joblib"))
    joblib.dump(seq_scaler, os.path.join(args.out_dir, "lstm_sequence_scaler.joblib"))
    joblib.dump(calibrator, os.path.join(args.out_dir, "lstm_calibrator.joblib"))
    print(f"[write] {args.out_dir}/lstm_model.keras (+ preprocessor/scaler/calibrator .joblib files)")

    write_lstm_predictions(args.out_dir, test, proba_test_raw, proba_test_calibrated, "lstm_churn_v1")
    update_metrics_json(args.out_dir, metrics_raw, metrics_calibrated)

    # --- 9. Permutation importance (labelled explicitly as NOT SHAP) ---
    print("\n[importance] Running permutation importance "
          "(model-agnostic stand-in — NOT SHAP, NOT directly comparable "
          "in magnitude to shap_global_importance.csv, only in rank order)...")
    importance_df = run_permutation_importance(
        model, X_seq_test, test[static_cols], y_test, static_cols
    )
    importance_path = os.path.join(args.out_dir, "lstm_permutation_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"[write] {importance_path}")
    print(importance_df.to_string(index=False))

    # --- 10. Compare to current champion; promote only if LSTM genuinely wins ---
    final_metrics = metrics_calibrated if metrics_calibrated else metrics_raw
    final_proba = proba_test_calibrated if proba_test_calibrated is not None else proba_test_raw
    feature_list = SEQ_CHANNELS + list(static_preprocessor.get_feature_names_out())
    maybe_promote_champion(args.out_dir, "lstm_churn_v1", "LSTM (hybrid sequence+static)",
                            final_metrics, feature_list, test, final_proba)

    print(f"\n[done] LSTM finished. Compare metrics.json's 'lr' / 'xgboost' / 'lstm_calibrated' "
          f"entries side by side for the report.")


if __name__ == "__main__":
    main()
