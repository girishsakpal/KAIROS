# Kairos — Phase 3 Findings & Status (19 Sep 2026)

Produced from the repo state in `kairos_zip.zip` (XGBoost `xgb_churn_v1`, 1,419-row time-based test set).
Everything below was recomputed from the files in the repo; XGBoost/SHAP could not be re-run in my sandbox,
so SHAP statements use your `shap_global_importance.csv` and the per-customer `top_features` as-is.

## 1. Phase 3 checklist (from the master doc, Section 9.1)

| Item | Status |
|---|---|
| PostgreSQL schema | Done (`db/schema.sql`); end-to-end load still unverified |
| Synthetic data conditioned on churn | Done |
| Leakage-safe features + time split | Done |
| LR + XGBoost + SHAP | Done, **with the calibration caveat in Section 3** |
| 3–4 written business insights (Section 4.1 style) | Drafted below (Section 4) |
| One Power BI ops dashboard on real predictions | **Data ready, dashboard to build** — `dashboards/powerbi/` |
| Skeleton Flask API (predict, customer-detail) | Not started |

## 2. Diagnostics that were open in the handoff — now closed

- **Leakage check on real Telco data:** best single-feature AUC is 0.74 (`tenure_months`), then 0.73, 0.67, 0.67.
  Nothing near the 0.86–0.96 seen in the buggy first generator. No gross leakage.
- **`months_of_usage_history`** is the #1 SHAP feature (0.83) but it is redundant with tenure: dropping it moves
  LR test AUC from 0.9004 to 0.9013 (no loss). Its gap to `tenure_months` also differs by class
  (−32.6 months for active, −12.2 for churned) because of the snapshot logic. Harmless for accuracy, but it
  inflates the SHAP ranking. **Recommend dropping it.**
- **`plan_type` is a 1:1 relabel of `contract_type`** (Basic = Month-to-month, Standard = One year,
  Pro = Two year, zero exceptions). SHAP credit is being split between two names for one signal. **Recommend dropping `plan_type`.**
- **SHAP vs the Section 4.1 narrative:** failed payments ranks 6th (0.37), behind the tenure family, plan type,
  monthly charge and feature adoption (0.43). So "failed payments is the single strongest driver" is **not** supported;
  see the rewritten insight below.

## 3. Calibration problem — please read before using Brier or revenue numbers

Mean predicted churn is **0.565** but actual churn in the test set is **0.409**. Decile view:

| Decile (1 = riskiest) | Avg predicted | Actual churn |
|---|---|---|
| 1 | 0.995 | 0.993 |
| 3 | 0.918 | 0.768 |
| 5 | 0.710 | 0.394 |
| 7 | 0.359 | 0.190 |
| 10 | 0.029 | 0.000 |

Ranking is excellent (decile 1 captures 24% of churners, deciles 1–3 capture 65%), but the probabilities are
systematically too high. Cause: both models are trained with class re-weighting (`class_weight="balanced"` for LR,
`scale_pos_weight` for XGBoost), which pushes probabilities up by design.

Consequences:
- The README line saying LR's Brier 0.25 means it is "closer to random guessing" is **wrong**; LR ranks well (AUC 0.90),
  its probabilities are just inflated by the weighting. "XGBoost is 36% better calibrated" is therefore not a clean
  finding yet. A constant base-rate predictor scores 0.242, so neither model is well calibrated.
- Anything of the form "expected revenue = charge × probability" will be overstated. The dashboard data therefore
  carries `monthly_revenue_exposed` (independent of calibration) as the primary figure.
- Test prevalence is 41% vs 23% in train and 26.5% overall. This is a side-effect of the synthetic snapshot-date
  logic in the time split. PR-AUC depends on prevalence, so quote it against the 0.41 baseline, not 0.5.

**Fix (needs your machine, since XGBoost isn't installable in my sandbox):** either train without the re-weighting,
or keep it and fit a calibrator (isotonic or Platt) on a validation slice carved from the *end of train*; then report
Brier before/after and plot a reliability curve. That closes the "calibration curve" open item honestly.

## 4. Insights in Section 4.1 style (all from full 7,043-row feature set unless noted)

**Driver.** In the last 90 days, customers with 0 failed payments churn at 18.5%, with 1 at 48.1%, with 2+ at ~87%
(n = 275). By SHAP it is the second most important *actionable* behaviour after feature adoption (0.37 vs 0.43).

**Cohort.** Month-to-month customers with at least one failed payment in 90 days churn at 71.6% (n = 1,047),
about 2.7× the 26.5% base rate; month-to-month with no failed payment is 32.0%. New customers (tenure ≤ 12 months)
paying by electronic check churn at 62.0% (n = 978), 2.3× base.

**Segment recommendation.**
- Customers in their first 6 months churn at 52.9% vs 9.5% for 49+ months, so onboarding contact belongs in the first half-year.
- Electronic-check payers churn at 45.3% vs 15–19% for automatic or mailed-check payers, so a switch-to-auto-pay
  offer is the cheapest lever.
- Customers in the bottom quartile of usage trend churn at 38.5% vs 20–25% elsewhere, so declining usage is a trigger for outreach.
- The master doc's "outreach at day 45" example is **not supported**: there is no time-to-event analysis yet. Don't claim it.

**Business impact (illustrative, test set of 1,419 customers).** Deciles 1–3 (426 customers) carry $32.9k of monthly
revenue, and 88.7% of them actually churned. With an assumed 25% save rate, that is roughly $7.3k/month retained.
Currency is USD because the Telco seed is USD.

## 5. Viva framing you should be ready for

The usage, ticket and transaction tables are **synthetic and generated conditioned on the churn label**, so behavioural
findings (failed payments, usage trend, adoption) reflect the generator's assumptions plus the real Telco columns
(contract, tenure, payment method, charges). The contract/tenure/payment-method findings are the ones grounded in real data.
Say this before a panelist asks. The 0.90 AUC is also higher than the usual ~0.84 on raw Telco because the synthetic
behavioural features add signal by construction.

## 6. Repo hygiene

- `.env` is inside the zip. `OPENAI_API_KEY` is empty right now, but keep `.env` out of git and out of shared zips (check `.gitignore` covers it).
- `synthetic_data/` and `telco dataset/` are byte-identical stale copies of `data/synthetic/` and `data/raw/`. Delete them.
- `mlflow.db` (0.9 MB) and the `.joblib` files are fine locally but shouldn't be committed.

## 7. Suggested order from here

1. Drop `months_of_usage_history` and `plan_type`, add calibration, re-run training (Section 3), re-run `prepare_dashboard_data.py`.
2. Build the Power BI ops dashboard (`dashboards/powerbi/POWERBI_BUILD_GUIDE.md`).
3. Load into Postgres and confirm the JSONB `top_features` parses.
4. Skeleton Flask API: `/predict` and `/customer-detail` per `docs/Kairos_API_Contracts.md`.
