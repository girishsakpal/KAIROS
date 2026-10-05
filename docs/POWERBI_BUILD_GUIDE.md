# Power BI Ops Dashboard — Build Guide

Data is prepared. Power BI Desktop (Windows) is needed to build the `.pbix`; save it as `dashboards/powerbi/kairos_ops.pbix`.

**Refresh the data any time:** `python3 dashboards/powerbi/prepare_dashboard_data.py`

## 1. Load data
Home → Get data → Text/CSV → load all three from `dashboards/powerbi/data/`:
`at_risk_customers`, `decile_lift`, `shap_global`. No relationships are needed (flat tables).
Set `risk_tier` sort order: create a column `Tier Order = SWITCH([risk_tier],"High",1,"Medium",2,"Low",3)`, then Sort by column.

## 2. Measures (on `at_risk_customers`)
```DAX
Customers Scored = COUNTROWS(at_risk_customers)
High Risk Customers = CALCULATE([Customers Scored], at_risk_customers[risk_tier] = "High")
Avg Churn Probability = AVERAGE(at_risk_customers[churn_probability])
Revenue Exposed (High) = CALCULATE(SUM(at_risk_customers[monthly_revenue_exposed]), at_risk_customers[risk_tier] = "High")
Actual Churn Rate = AVERAGE(at_risk_customers[actual_churn])
High-tier Precision = CALCULATE([Actual Churn Rate], at_risk_customers[risk_tier] = "High")

Save Rate = GENERATESERIES(0.05, 0.50, 0.05)          -- creates a table; use as a slicer
Save Rate Value = SELECTEDVALUE('Save Rate'[Value], 0.25)
Illustrative Revenue Saved = [Revenue Exposed (High)] * [High-tier Precision] * [Save Rate Value]
```
Use `monthly_revenue_exposed`, not `expected_monthly_revenue_at_risk`, for headline numbers (probabilities are currently inflated; see `docs/Phase3_Findings.md` §3).

## 3. Pages

**Page 1 — Operations overview**
- Cards: Customers Scored, High Risk Customers, Revenue Exposed (High), Avg Churn Probability.
- Slicers: region, risk_tier, contract_type, tenure_band, recommended_action, signup_channel.
- Table: risk_rank, customer_id, churn_probability (data bars), risk_tier, driver_1, driver_2, driver_3, recommended_action. Sort by risk_rank.
- Bar chart: count of customers by recommended_action, split by risk_tier (agent workload view).
- Bar chart: High Risk Customers by region.

**Page 2 — Customer drill-through**
- Right-click drill-through field: customer_id. Show probability gauge, plan/contract/charge/tenure, and a small table of
  driver_1..3 with `driver_n_direction` and `driver_n_shap`. Add a text box: recommended_action.

**Page 3 — Model validation (for the viva)**
- Clustered column: `decile_lift[actual_churn_rate]` and `decile_lift[avg_predicted]` by risk_decile. The gap between the two
  bars is the calibration issue, so show it deliberately until recalibrated.
- Line: cum_churners_captured_pct by risk_decile (gains curve). Card: lift of decile 1 (2.4×).
- Bar: `shap_global` top 10 by mean_abs_shap, coloured by group.

**Page 4 (optional) — What-if**: Save Rate slicer + Illustrative Revenue Saved card.

## 4. Notes
- `actual_churn` exists only because this is a held-out test set; in production that column would not exist. Keep it on Page 3 only.
- Theme: save a JSON theme into `dashboards/powerbi/` for reuse in Phase 6.
- After you drop `months_of_usage_history`/`plan_type` and retrain, re-run the prep script and hit Refresh.
