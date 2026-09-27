# Kairos — LSTM Model Notes

`src/models/train_lstm.py` adds a third model: a hybrid LSTM (usage sequence) +
dense (static tabular) network, alongside LR and XGBoost.

## What was verified, what wasn't
TensorFlow isn't installed in my sandbox (no network access), so — same caveat
as XGBoost/SHAP/MLflow in the original handoff — the Keras model itself has
**not** been run. What I did verify directly, using your real repo data:
- Data loading, redundant-column dropping (`plan_type`, `months_of_usage_history`,
  the six usage-aggregate columns) — correct, matches `Phase3_Findings.md`.
- Sequence building respects each customer's own `snapshot_date` cutoff — no
  leakage. Sequence lengths across your real train set: 515 customers with 3
  valid months, 1,246 with 4, 1,752 with 5, 2,111 with 6. None with zero.
- Padding/masking: every padded position after scaling is exactly `-10.0` (the
  masking sentinel), and no real scaled value ever lands there — checked
  programmatically, not just by eye.
- Static preprocessor builds a clean 32-column matrix with no errors.

What I could **not** verify: model training actually converging, Keras's
`validation_split` behaving as documented (it should take the tail of the
array, which is why train is pre-sorted by `snapshot_date`), and the
permutation-importance loop's runtime (it re-runs `model.predict` once per
static column — with ~17 columns × 3 repeats + one sequence pass, expect this
to take a few minutes on CPU for ~1,400 test rows, not more).

## How to run it
```bash
pip install tensorflow --break-system-packages   # CPU build is fine at this scale
python3 src/models/train_lstm.py
```
Uses `data/processed/features_{train,test}.csv` and `data/synthetic/usage_logs.csv`
by default — same as your other Phase 3 scripts.

## What to check when you run it
1. **Does it error?** Most likely failure point if any: the `Masking` /
   `LSTM` layer shapes, or a Keras version difference in how `class_weight`
   interacts with a multi-input functional model. If it errors there, paste
   me the traceback.
2. **Training curve** — does `val_auc` improve then plateau (normal) or stay
   flat near 0.5 (would suggest the sequence branch isn't learning anything,
   possibly because 3–6 timesteps is too short for an LSTM to have an edge
   over just averaging — a legitimate, reportable finding either way).
3. **Calibration** — check the printed "mean predicted before/after
   calibration" line. If it's still far from the actual test churn rate after
   calibration, the validation slice may be too small or too different from
   test; flag it and I'll adjust `VALIDATION_FRACTION`.
4. **Does LSTM beat XGBoost?** On this dataset, I'd genuinely expect it not
   to, or to land close: XGBoost already gets the same usage information via
   6 hand-built summary features, tree ensembles are typically strong on
   small tabular datasets, and 3–6 timesteps is a short sequence for an LSTM
   to have much room to add value. **A result where LSTM doesn't win is not
   a bug** — it's a defensible, citable finding for your report ("sequence
   modeling did not outperform gradient-boosted trees on this
   snapshot-heavy dataset, consistent with tree ensembles' typical edge on
   small tabular problems"). The script only promotes LSTM to champion if it
   genuinely wins on test ROC-AUC — it won't silently overwrite your existing
   XGBoost results either way.

## What it produces
Everything under `models/`:
- `lstm_model.keras`, `lstm_static_preprocessor.joblib`, `lstm_sequence_scaler.joblib`, `lstm_calibrator.joblib`
- `lstm_predictions_test.csv` — raw and calibrated probability side by side
- `lstm_permutation_importance.csv` — **not SHAP**, a model-agnostic stand-in (shuffle-and-measure-AUC-drop); comparable in rank order to `shap_global_importance.csv`, not in magnitude
- `metrics.json` updated in place (adds `lstm` / `lstm_calibrated`, keeps your existing `logistic_regression` / `xgboost` entries — doesn't overwrite the whole file)
- `model_registry_row.json` / `churn_predictions_test.csv` — **only overwritten if LSTM beats the current champion's test ROC-AUC**; the script always prints the comparison either way

## requirements.txt
Add one line under `# ML / XAI`:
```
tensorflow>=2.15
```
