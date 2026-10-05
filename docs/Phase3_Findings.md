# Kairos — Phase 3 Findings & Status (updated 5 Oct 2026, after `--refit-full` run)

Supersedes the earlier 5 Oct draft. Model numbers come from your terminal output of
`train_baseline.py --refit-full` and `prepare_dashboard_data.py`, plus `metrics.json` from the
LSTM run. Test set = 1,419 customers, time-based split, 40.9% churn.

**Which XGBoost is served:** the run used `--refit-full`, so the dashboard, the API and the SHAP
file all come from the XGBoost model **refit on the full train set**, with the isotonic calibrator
fit on the validation slice. Per-feature SHAP statements use that model's
`shap_global_importance.csv` (37 encoded features).

## 1. Phase 3 checklist (master doc, Section 9.1)

| Item | Status |
|---|---|
| PostgreSQL schema | Done (`db/schema.sql`); end-to-end load still unverified |
| Synthetic data conditioned on churn | Done |
| Leakage-safe features + time split | Done |
| LR + XGBoost + SHAP | Done: redundant features dropped, calibrated, refit on full train (Step 5) |
| LSTM comparison model | Done (`train_lstm.py`) |
| 3–4 written business insights (Section 4.1 style) | Drafted and re-checked against v2 SHAP (Section 5 below) |
| Power BI ops dashboard on real predictions | Done (`kairos_ops.pbix`, 4 pages); **Refresh needed** with the refit data |
| Skeleton Flask API (predict, customer-detail) | Done: 4 endpoints, 19 tests pass on your machine; no auth, CSV-backed |
| Postgres load | Not started |

## 2. Diagnostics from the 19 Sep version — now actioned

- `plan_type` was a 1:1 relabel of `contract_type`; `months_of_usage_history` was redundant with
  tenure and inflated the SHAP ranking. **Both are now dropped** from LR and XGBoost (the LSTM
  already dropped them).
- Leakage check: best single-feature AUC was 0.74 (`tenure_months`). No gross leakage.

## 3. Model comparison (test set)

| Model | ROC-AUC | PR-AUC | Brier | Mean predicted churn (actual 0.409) |
|---|---|---|---|---|
| Logistic Regression, fit slice, raw | 0.8993 | 0.8725 | 0.1309 | 0.468 |
| Logistic Regression, fit slice, calibrated | 0.8971 | 0.8553 | 0.1290 | 0.430 |
| Logistic Regression, refit, raw | 0.9015 | 0.8759 | 0.1335 | 0.487 |
| Logistic Regression, refit, calibrated | 0.9006 | 0.8637 | 0.1285 | 0.450 |
| XGBoost, fit slice, raw | 0.8853 | 0.8516 | 0.1403 | 0.468 |
| XGBoost, fit slice, calibrated | 0.8833 | 0.8386 | 0.1389 | 0.453 |
| **XGBoost, refit, raw** | 0.8961 | 0.8670 | 0.1329 | 0.466 |
| **XGBoost, refit, calibrated (served)** | **0.8936** | **0.8527** | **0.1328** | **0.451** |
| LSTM (hybrid), raw | 0.9174 | 0.8957 | 0.1414 | — |
| LSTM (hybrid), calibrated | 0.9158 | 0.8851 | 0.1137 | 0.421 |

Before Step 5 (v1, uncalibrated, with the redundant features): LR 0.8984 / 0.8728 / 0.2498 and
XGBoost 0.9047 / 0.8824 / 0.1599, with mean predicted churn 0.565.

A model that always predicts the base rate scores a Brier of about 0.242 on this test set.

**What the numbers say**

1. **Calibration largely worked.** The mean-predicted gap fell from +0.156 (0.565 vs 0.409) to
   +0.042 for the served XGBoost. Its Brier improved 0.160 → 0.133, and LR's 0.250 → 0.129.
2. **Calibration is close for most deciles but not all.** Per-decile gaps (predicted minus actual)
   are within about 0.03 for deciles 1, 3, 4, 7, 8, 9 and 10, but decile 2 over-predicts by 0.10
   (0.948 vs 0.852) and deciles 5 and 6 by 0.10 and 0.14. The calibrator was fit on a validation
   slice with 43.9% churn, above the test rate. Treat that as a likely contributor, not a
   confirmed cause.
3. **The refit recovered about half of XGBoost's accuracy loss.** Calibrated AUC went
   0.8833 → 0.8936 (+0.010); v1 was 0.9047. The calibration holdout therefore did cost
   accuracy, because it removed the most recent, most test-like rows from training. Calibration
   did not degrade: mean predicted stayed at 0.451 (it drifted up for LR, 0.430 → 0.450).
4. **About 0.011 AUC is still unexplained.** Candidates, all untested: the two dropped features
   and the absence of hyperparameter tuning. The v1 comparison also changes the feature set, so
   do not say the refit "fixed" the drop.
5. **XGBoost is still the weakest of the three on ROC-AUC**: LSTM 0.9158 > LR 0.9006 > XGBoost
   0.8936. This contradicts the plan's framing of XGBoost as the main model. It is a legitimate
   finding for the report; do not hide it.
6. **LSTM is the champion** by ROC-AUC (0.9158) and Brier (0.1137). The sequence branch appears
   to carry real signal beyond the six hand-built usage features. Caveats: it used its own
   validation split (Keras `validation_split`), and it is compared against baselines whose
   calibrator was fit on a different slice, so the calibration comparison is approximate.

**Why the dashboard uses XGBoost even though the LSTM is champion:** the LSTM has no per-customer
explanation (permutation importance is global only), and the dashboard's drivers and recommended
actions need per-customer SHAP. This is a deliberate split — the LSTM scores best, XGBoost explains
— and it should be stated in the viva before a panelist spots it. A real weakness follows from
it: the dashboard explains the weakest model. Logistic Regression beats XGBoost on AUC and has
native linear explanations, so using LR (with linear SHAP) for the explanation layer is a
defensible alternative if the panel pushes on this.

## 4. Ranking quality (XGBoost refit, the dashboard model)

| Decile (1 = riskiest) | Avg predicted | Actual churn | Lift | Cum. churners captured |
|---|---|---|---|---|
| 1 | 0.994 | 0.972 | 2.37 | 23.8% |
| 2 | 0.948 | 0.852 | 2.08 | 44.6% |
| 3 | 0.762 | 0.775 | 1.89 | 63.5% |
| 4 | 0.550 | 0.542 | 1.32 | 76.8% |
| 5 | 0.485 | 0.380 | 0.93 | 86.1% |
| 6 | 0.382 | 0.246 | 0.60 | 92.1% |
| 7 | 0.178 | 0.169 | 0.41 | 96.2% |
| 8 | 0.140 | 0.113 | 0.28 | 99.0% |
| 9 | 0.052 | 0.035 | 0.09 | 99.8% |
| 10 | 0.013 | 0.007 | 0.02 | 100% |

Headline numbers for the viva: **decile-1 lift 2.37x** (v1 was 2.43x) and **top 3 deciles capture
63.5% of churners** (v1 was about 65%). Any slide or note quoting 2.43x and 65% is stale, and so
is anything quoting 61.8% from the earlier fit-slice-only run.

Global SHAP by business group (sum of mean |SHAP|): Engagement 1.476, Plan & Billing 1.184,
Payments 1.007, Tenure 0.604, Support 0.565, Demographic 0.460 (total 5.295). Tenure fell from the
top group once `months_of_usage_history` was removed, as expected. Demographic includes `age`,
which is non-actionable and is ignored when choosing the recommended action. Because the model
changed, the recommended-action mix changed too (for example "Escalate to retention specialist"
is now 108 customers, down from 133 in v1); re-check the Page 1 workload chart after Refresh.

Top features by mean |SHAP| (XGBoost refit):

| Rank | Feature | Mean abs SHAP |
|---|---|---|
| 1 | Month-to-month contract | 0.607 |
| 2 | Customer tenure | 0.604 |
| 3 | Monthly charge | 0.437 |
| 4 | Feature adoption (recent) | 0.433 |
| 5 | Failed payments (90d) | 0.389 |
| 6 | Usage trend (session count) | 0.350 |
| 7 | Ticket resolution time (90d) | 0.272 |
| 8 | Days since last active | 0.245 |
| 9 | Session minutes (recent) | 0.230 |
| 10 | Session count (recent) | 0.219 |

Contract, tenure and charge (real Telco columns) take the top three places. The contract signal is
carried by one feature (month-to-month) since `plan_type` was dropped.

**Rank stability caveat.** Between the fit-slice model and the refit model, failed payments moved
from 6th (0.334) to 5th (0.389) and usage trend from 5th (0.344) to 6th (0.350). Ranks 4–6 are
within about 0.08 of each other and swap between fits, so quote them as "a top-six driver", not as
a precise order. Contract and tenure at the top were stable.

Caveat on rank 7: ticket resolution time ranks high partly because the generator gives
elevated-risk customers longer resolution times (up to 84h vs 48h). That is a generator
assumption, not a real-world finding. Do not present it as a discovered driver.

## 5. Business insights (Section 4.1 style)

The percentages below are computed from the 7,043-row feature set and do not depend on the model
version, so they stand. SHAP-rank statements use the v2 model.

**Driver.** In the last 90 days, customers with 0 failed payments churn at 18.5%, with 1 at 48.1%,
with 2+ at about 87% (n = 275). On the refit model, failed payments is the **second** most
important *actionable* behaviour by SHAP (0.389), behind feature adoption (0.433), and 5th overall;
usage trend is next (0.350). The order among these three is not stable across fits (see Section 4),
so say "a top-five driver with the cleanest threshold effect", not "the strongest driver". The master
doc's "single strongest driver" wording remains unsupported.

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

**Business impact (illustrative).** Deciles 1–3 are 426 customers, and about 87% of them actually
churned (0.972, 0.852 and 0.775 averaged over equal-size deciles; cross-check: 63.5% of about 580
churners is about 368, and 368 / 426 = 86%). Read the revenue figure for those 426 straight off the
refreshed dashboard (`monthly_revenue_exposed`). The earlier $32.9k / 88.7% / $7.3k figures were for
v1 and must be replaced. Keep the 25% save rate labelled as an assumption.

## 6. Viva framing

- The usage, ticket and transaction tables are synthetic and conditioned on the churn label, so
  behavioural findings reflect the generator's assumptions. Contract, tenure and payment-method
  findings are grounded in real Telco data. Say this first.
- The 0.88–0.92 AUCs are above the usual ~0.84 on raw Telco because the synthetic behavioural
  features add signal by construction.
- Test prevalence (41%) is far above train-fit prevalence (19%). This is a side effect of the
  synthetic snapshot-date logic, not a real distribution shift. Quote PR-AUC against the 0.41
  baseline.
- Say plainly that XGBoost, the explainable model, is not the most accurate model here (LSTM 0.916,
  LR 0.901, XGBoost 0.894 ROC-AUC) and explain the split: the LSTM scores, XGBoost explains.
- Feature rankings 4–6 are close and swap between fits. Do not defend a precise order.

## 7. Repo hygiene

- Keep `.env`, `mlflow.db` and the `.joblib` files out of git and out of shared zips.
- `README.md` still shows the v1 LR/XGBoost numbers in its results block. Update it from
  `models/metrics.json`.

## 8. Suggested order from here

1. Refresh `kairos_ops.pbix` (Home → Refresh). Re-check Page 3 (calibration gap is small for most deciles;
   deciles 2, 5 and 6 still over-predict by 0.10–0.14), the Decile 1 Lift card (2.37x) and the Page 1 workload chart.
2. Load into Postgres and confirm the JSONB `top_features` parses; then swap `CsvRepository` for a
   Postgres-backed one (the API's only storage-touching file).
3. `git add` / `git commit` the Step 5 changes, the API, the `.pbix` and this file. Keep `.env`,
   `mlflow.db` and `.joblib` files out of git.
4. Update `README.md`'s results block from `models/metrics.json` (still shows v1 numbers).
5. Optional, if time allows: tune XGBoost with Optuna (the remaining ~0.011 AUC gap), or test an
   LR-with-linear-SHAP explanation layer.