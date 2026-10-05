# Kairos — Phase 3 Findings & Status (updated 5 Oct 2026)

Supersedes the 19 Sep version. Model numbers below come from your terminal output of
`train_baseline.py` and `prepare_dashboard_data.py` (Step 5 run), plus `metrics.json` from the
LSTM run. Test set = 1,419 customers, time-based split, 40.9% churn.

Per-feature SHAP statements use the v2 `shap_global_importance.csv` (37 encoded features).

## 1. Phase 3 checklist (master doc, Section 9.1)

| Item | Status |
|---|---|
| PostgreSQL schema | Done (`db/schema.sql`); end-to-end load still unverified |
| Synthetic data conditioned on churn | Done |
| Leakage-safe features + time split | Done |
| LR + XGBoost + SHAP | Done, redundant features dropped, calibrated (Step 5) |
| LSTM comparison model | Done (`train_lstm.py`) |
| 3–4 written business insights (Section 4.1 style) | Drafted and re-checked against v2 SHAP (Section 5 below) |
| Power BI ops dashboard on real predictions | Done (`kairos_ops.pbix`, 4 pages); needs Refresh after Step 5 data |
| Skeleton Flask API (predict, customer-detail) | Not started |

## 2. Diagnostics from the 19 Sep version — now actioned

- `plan_type` was a 1:1 relabel of `contract_type`; `months_of_usage_history` was redundant with
  tenure and inflated the SHAP ranking. **Both are now dropped** from LR and XGBoost (the LSTM
  already dropped them).
- Leakage check: best single-feature AUC was 0.74 (`tenure_months`). No gross leakage.

## 3. Model comparison (test set)

| Model | ROC-AUC | PR-AUC | Brier | Mean predicted churn (actual 0.409) |
|---|---|---|---|---|
| Logistic Regression, raw | 0.8993 | 0.8725 | 0.1309 | 0.468 |
| Logistic Regression, calibrated | 0.8971 | 0.8553 | 0.1290 | 0.430 |
| XGBoost, raw | 0.8853 | 0.8516 | 0.1403 | 0.468 |
| XGBoost, calibrated | 0.8833 | 0.8386 | 0.1389 | 0.453 |
| LSTM (hybrid), raw | 0.9174 | 0.8957 | 0.1414 | — |
| LSTM (hybrid), calibrated | 0.9158 | 0.8851 | 0.1137 | 0.421 |

Before Step 5 (v1, uncalibrated, with the redundant features): LR 0.8984 / 0.8728 / 0.2498 and
XGBoost 0.9047 / 0.8824 / 0.1599, with mean predicted churn 0.565.

A model that always predicts the base rate scores a Brier of about 0.242 on this test set.

**What the numbers say**

1. **Calibration largely worked.** The mean-predicted gap fell from +0.156 (0.565 vs 0.409) to
   +0.044 for XGBoost. XGBoost Brier improved 0.160 → 0.139 and LR 0.250 → 0.129, so neither is
   near the 0.242 base-rate score any more.
2. **Calibration is not perfect.** XGBoost still over-predicts in the middle deciles (decile 6:
   predicted 0.399, actual 0.275; decile 5: 0.488 vs 0.408). The calibrator was fit on a
   validation slice with 43.9% churn, a little above the test rate. Treat that as a likely
   contributor, not a confirmed cause.
3. **Ranking cost for XGBoost.** Calibrated AUC is 0.8833 vs 0.9047 before, a drop of about 0.02.
   LR barely moved (0.8984 → 0.8993 raw), which matches the earlier ablation. Candidate causes,
   all untested: 15% less training data, the dropped features, and no hyperparameter tuning.
4. **XGBoost is now the weakest of the three on ROC-AUC**, behind LR and the LSTM. This
   contradicts the plan's framing of XGBoost as the main model. It is a legitimate finding for
   the report, and it should not be hidden.
5. **LSTM is the champion** by ROC-AUC (0.9158 calibrated) and by Brier (0.1137). The sequence
   branch appears to carry real signal beyond the six hand-built usage features. Caveat: the
   LSTM used its own validation split (Keras `validation_split`), so its calibration slice is
   not identical to the one used for LR and XGBoost.

**Why the dashboard uses XGBoost even though the LSTM is champion:** the LSTM has no per-customer
explanation (permutation importance is global only), and the dashboard's drivers and recommended
actions need per-customer SHAP. This is a deliberate split — the LSTM scores best, XGBoost explains
— and it should be stated in the viva before a panelist spots it. A real weakness follows from
it: the dashboard explains the weakest model.

## 4. Ranking quality (XGBoost v2, the dashboard model)

| Decile (1 = riskiest) | Avg predicted | Actual churn | Lift | Cum. churners captured |
|---|---|---|---|---|
| 1 | 0.992 | 0.972 | 2.37 | 23.8% |
| 2 | 0.943 | 0.824 | 2.01 | 43.9% |
| 3 | 0.756 | 0.732 | 1.79 | 61.8% |
| 4 | 0.546 | 0.535 | 1.31 | 74.9% |
| 5 | 0.488 | 0.408 | 1.00 | 84.9% |
| 6 | 0.399 | 0.275 | 0.67 | 91.6% |
| 7 | 0.182 | 0.176 | 0.43 | 95.9% |
| 8 | 0.143 | 0.085 | 0.21 | 97.9% |
| 9 | 0.060 | 0.085 | 0.21 | 100% |
| 10 | 0.017 | 0.000 | 0.00 | 100% |

Headline numbers for the viva: **decile-1 lift 2.37x** (was 2.43x in v1) and **top 3 deciles
capture 61.8% of churners** (was about 65%). Any slide or note that quotes 2.43x and 65% is now stale.

Global SHAP by business group (sum of mean |SHAP|): Engagement 1.517, Plan & Billing 1.167,
Payments 1.015, Support 0.634, Tenure 0.626, Demographic 0.566. Tenure fell from the top group
once `months_of_usage_history` was removed, as expected. Demographic includes `age`, which is
non-actionable and is ignored when choosing the recommended action.

Top features by mean |SHAP| (XGBoost v2):

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | Customer tenure | 0.626 |
| 2 | Month-to-month contract | 0.574 |
| 3 | Monthly charge | 0.480 |
| 4 | Feature adoption (recent) | 0.414 |
| 5 | Usage trend (session count) | 0.344 |
| 6 | Failed payments (90d) | 0.334 |
| 7 | Days since last active | 0.314 |
| 8 | Ticket resolution time (90d) | 0.273 |
| 9 | Session minutes (recent) | 0.224 |
| 10 | Session count (recent) | 0.222 |

Contract, tenure and charge (real Telco columns) take the top three places. The contract signal is
now carried by one feature (month-to-month) instead of being shared with `plan_type`. Failed
payments stays 6th (0.334, was 0.37).

Caveat on rank 8: ticket resolution time ranks high partly because the generator gives
elevated-risk customers longer resolution times (up to 84h vs 48h). That is a generator
assumption, not a real-world finding. Do not present it as a discovered driver.

## 5. Business insights (Section 4.1 style)

The percentages below are computed from the 7,043-row feature set and do not depend on the model
version, so they stand. SHAP-rank statements use the v2 model.

**Driver.** In the last 90 days, customers with 0 failed payments churn at 18.5%, with 1 at 48.1%,
with 2+ at about 87% (n = 275). On the v2 model, failed payments is the **third** most important
*actionable* behaviour by SHAP (0.334), behind feature adoption (0.414) and usage trend (0.344), and
6th overall. The earlier "second" ranking (0.37 vs 0.43) was the v1 model. "Single strongest driver"
is not supported; "a top-six driver and the cleanest threshold effect" is.

**Cohort.** Month-to-month customers with at least one failed payment in 90 days churn at 71.6%
(n = 1,047), about 2.7x the 26.5% base rate; month-to-month with no failed payment is 32.0%. New
customers (tenure <= 12 months) paying by electronic check churn at 62.0% (n = 978), 2.3x base.

**Segment recommendation.**
- First 6 months: 52.9% churn vs 9.5% for 49+ months, so onboarding contact belongs in the first
  half-year.
- Electronic-check payers churn at 45.3% vs 15–19% for automatic or mailed-check payers, so a
  switch-to-auto-pay offer is the cheapest lever.
- Bottom quartile of usage trend: 38.5% vs 20–25% elsewhere, so declining usage is an outreach
  trigger.
- The master doc's "outreach at day 45" example is **not supported**: there is no time-to-event
  analysis yet. Do not claim it.

**Business impact (illustrative).** Re-compute from `at_risk_customers.csv` after Refresh: deciles
1–3 are now 426 customers carrying a monthly revenue figure you can read straight off the
dashboard (`monthly_revenue_exposed`), and about 84% of those 426 actually churned
(0.972, 0.824 and 0.732 averaged over equal-size deciles; cross-check: 61.8% of ~580 churners is ~358, and 358 / 426 = 84%). The earlier $32.9k / 88.7% / $7.3k figures
were for v1 and must be replaced. Keep the 25% save rate labelled as an assumption.

## 6. Viva framing

- The usage, ticket and transaction tables are synthetic and conditioned on the churn label, so
  behavioural findings reflect the generator's assumptions. Contract, tenure and payment-method
  findings are grounded in real Telco data. Say this first.
- The 0.88–0.92 AUCs are above the usual ~0.84 on raw Telco because the synthetic behavioural
  features add signal by construction.
- Test prevalence (41%) is far above train-fit prevalence (19%). This is a side effect of the
  synthetic snapshot-date logic, not a real distribution shift. Quote PR-AUC against the 0.41
  baseline.
- Say plainly that XGBoost, the explainable model, is not the most accurate model here.

## 7. Repo hygiene

- Keep `.env`, `mlflow.db` and the `.joblib` files out of git and out of shared zips.
- `README.md` still shows the v1 LR/XGBoost numbers in its results block. Update it from
  `models/metrics.json`.

## 8. Suggested order from here

1. Refresh `kairos_ops.pbix` (Home → Refresh) and re-check Page 3 and the Decile 1 Lift card
   (now 2.37x).
2. Decide how to handle the XGBoost accuracy drop. Options: refit on the full train set after
   fitting the calibrator; tune with Optuna as planned; or use LR with linear SHAP for the
   explanation layer.
3. Skeleton Flask API: `/predict` and `/customer-detail` per `docs/Kairos_API_Contracts.md`.
4. Load into Postgres and confirm the JSONB `top_features` parses.
5. `git add` / `git commit` the Step 5 changes, the `.pbix` and this file.