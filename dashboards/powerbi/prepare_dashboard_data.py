"""
Build Power BI-ready tables from the Phase 3 model outputs.

Inputs  (all produced by earlier Phase 3 steps):
  models/churn_predictions_test.csv   - probabilities + top-3 SHAP drivers (JSON string)
  models/shap_global_importance.csv   - mean |SHAP| per feature
  data/synthetic/customers.csv        - region, channel, age, gender
  data/synthetic/subscriptions.csv    - plan, contract, monthly_charge, tenure

Outputs (dashboards/powerbi/data/):
  at_risk_customers.csv  - one row per scored customer, flat, plain-English drivers + action
  decile_lift.csv        - model validation page (lift / capture by risk decile)
  shap_global.csv        - global driver ranking with business-friendly labels and groups

Run from repo root:  python3 dashboards/powerbi/prepare_dashboard_data.py
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dashboards" / "powerbi" / "data"
OUT.mkdir(parents=True, exist_ok=True)

# feature -> (plain-English label, business group, retention action if it is a risk driver)
FEATURES = {
    "num__months_of_usage_history": ("Usage history length", "Tenure", "Early-life onboarding call"),
    "num__tenure_months": ("Customer tenure", "Tenure", "Early-life onboarding call"),
    "cat__plan_type_Basic": ("Basic plan (month-to-month)", "Plan & Billing", "Offer annual-contract incentive"),
    "cat__plan_type_Pro": ("Pro plan", "Plan & Billing", "Offer annual-contract incentive"),
    "cat__plan_type_Standard": ("Standard plan", "Plan & Billing", "Offer annual-contract incentive"),
    "cat__contract_type_Month-to-month": ("Month-to-month contract", "Plan & Billing", "Offer annual-contract incentive"),
    "cat__contract_type_One year": ("One-year contract", "Plan & Billing", "Offer annual-contract incentive"),
    "cat__contract_type_Two year": ("Two-year contract", "Plan & Billing", "Offer annual-contract incentive"),
    "num__monthly_charge": ("Monthly charge", "Plan & Billing", "Price-fit review / bundle offer"),
    "num__avg_feature_adoption_recent": ("Feature adoption", "Engagement", "Proactive feature walkthrough"),
    "num__avg_session_minutes_recent": ("Session minutes", "Engagement", "Proactive feature walkthrough"),
    "num__avg_session_count_recent": ("Session count", "Engagement", "Proactive feature walkthrough"),
    "num__session_count_trend": ("Usage trend", "Engagement", "Proactive check-in call"),
    "num__days_since_last_active": ("Days since last active", "Engagement", "Proactive check-in call"),
    "num__failed_payment_count_90d": ("Failed payments (90d)", "Payments", "Payment recovery + switch to auto-pay"),
    "num__failed_payment_ratio_90d": ("Failed-payment ratio", "Payments", "Payment recovery + switch to auto-pay"),
    "num__total_paid_90d": ("Amount paid (90d)", "Payments", "Payment recovery + switch to auto-pay"),
    "num__transaction_count_90d": ("Transaction count (90d)", "Payments", "Payment recovery + switch to auto-pay"),
    "cat__payment_method_Electronic check": ("Pays by electronic check", "Payments", "Payment recovery + switch to auto-pay"),
    "cat__payment_method_Mailed check": ("Pays by mailed check", "Payments", "Payment recovery + switch to auto-pay"),
    "cat__payment_method_Bank transfer (automatic)": ("Bank transfer payer", "Payments", "Payment recovery + switch to auto-pay"),
    "cat__payment_method_Credit card (automatic)": ("Credit card payer", "Payments", "Payment recovery + switch to auto-pay"),
    "num__ticket_count_90d": ("Support tickets (90d)", "Support", "Escalate to retention specialist"),
    "num__open_ticket_count": ("Open support tickets", "Support", "Escalate to retention specialist"),
    "num__avg_resolution_hours_90d": ("Ticket resolution time", "Support", "Escalate to retention specialist"),
    "num__billing_ticket_ratio_90d": ("Billing-ticket share", "Support", "Escalate to retention specialist"),
    "num__age": ("Customer age", "Demographic", "General retention check-in"),
}
def label(f):
    if f in FEATURES: return FEATURES[f]
    clean = f.split("__", 1)[-1].replace("_", " ")
    return (clean.capitalize(), "Demographic" if f.startswith("cat__") else "Other", "General retention check-in")

def parse_drivers(s):
    try: return json.loads(s)
    except Exception: return []

pred = pd.read_csv(ROOT / "models" / "churn_predictions_test.csv")
cust = pd.read_csv(ROOT / "data" / "synthetic" / "customers.csv")
subs = pd.read_csv(ROOT / "data" / "synthetic" / "subscriptions.csv")

df = (pred.merge(cust[["customer_id", "region", "signup_channel", "age", "gender"]], on="customer_id", how="left")
          .merge(subs[["customer_id", "plan_type", "contract_type", "monthly_charge", "tenure_months"]],
                 on="customer_id", how="left"))

# flatten top-3 SHAP drivers
# Action rule: follow the strongest RISK-RAISING driver, except:
#  - months_of_usage_history is redundant with tenure (ablation showed no AUC gain) -> ignored for actions
#  - age is not actionable -> ignored for actions
#  - tenure only triggers an onboarding action if the customer really is new (<= 12 months)
NOT_ACTIONABLE = {"num__months_of_usage_history", "num__age"}
rows = []
for s, tenure in zip(df["top_features"], df["tenure_months"]):
    d = parse_drivers(s)
    rec = {}
    action = "General retention check-in"
    action_set = False
    for i in range(3):
        if i < len(d):
            f = d[i]["feature"]
            lab, grp, act = label(f)
            rec[f"driver_{i+1}"] = lab
            rec[f"driver_{i+1}_group"] = grp
            rec[f"driver_{i+1}_direction"] = "Raises risk" if d[i]["direction"] == "increases_risk" else "Lowers risk"
            rec[f"driver_{i+1}_shap"] = d[i]["shap_value"]
            eligible = (d[i]["direction"] == "increases_risk" and f not in NOT_ACTIONABLE
                        and not (f == "num__tenure_months" and tenure > 12))
            if eligible and not action_set:
                action, action_set = act, True
        else:
            rec[f"driver_{i+1}"] = rec[f"driver_{i+1}_group"] = rec[f"driver_{i+1}_direction"] = None
            rec[f"driver_{i+1}_shap"] = None
    rec["recommended_action"] = action
    rows.append(rec)
df = pd.concat([df.drop(columns=["top_features"]), pd.DataFrame(rows)], axis=1)

df = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
df["risk_rank"] = df.index + 1
df["risk_decile"] = (df.index * 10 // len(df)) + 1          # 1 = highest-risk 10%
df["risk_tier"] = df["risk_tier"].str.capitalize()
# Monthly revenue tied to the customer (does NOT depend on probability calibration)
df["monthly_revenue_exposed"] = df["monthly_charge"]
# Probability-weighted version -- see calibration caveat in docs/Phase3_Findings.md before quoting this
df["expected_monthly_revenue_at_risk"] = (df["monthly_charge"] * df["churn_probability"]).round(2)
df["tenure_band"] = pd.cut(df["tenure_months"], [-1, 6, 12, 24, 48, 100],
                           labels=["0-6m", "7-12m", "13-24m", "25-48m", "49m+"]).astype(str)
df.to_csv(OUT / "at_risk_customers.csv", index=False)

# decile lift (model validation page)
base = df["actual_churn"].mean()
tot = df["actual_churn"].sum()
dec = (df.groupby("risk_decile")
         .agg(customers=("customer_id", "size"), avg_predicted=("churn_probability", "mean"),
              actual_churn_rate=("actual_churn", "mean"), churners=("actual_churn", "sum"),
              monthly_revenue=("monthly_charge", "sum"))
         .reset_index())
dec["lift"] = dec["actual_churn_rate"] / base
dec["cum_churners_captured_pct"] = dec["churners"].cumsum() / tot
dec.round(4).to_csv(OUT / "decile_lift.csv", index=False)

# global SHAP with labels
g = pd.read_csv(ROOT / "models" / "shap_global_importance.csv")
g["label"] = g["feature"].map(lambda f: label(f)[0])
g["group"] = g["feature"].map(lambda f: label(f)[1])
g.sort_values("mean_abs_shap", ascending=False).to_csv(OUT / "shap_global.csv", index=False)

print(f"at_risk_customers.csv : {len(df)} rows | base churn {base:.3f}")
print(dec[["risk_decile", "customers", "avg_predicted", "actual_churn_rate", "lift", "cum_churners_captured_pct"]].round(3).to_string(index=False))
print(df.groupby("recommended_action").size().sort_values(ascending=False).to_string())
print(g.groupby("group").mean_abs_shap.sum().sort_values(ascending=False).round(3).to_string())
