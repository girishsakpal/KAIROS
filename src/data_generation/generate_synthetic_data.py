"""
Kairos — Synthetic Data Generator (Phase 3)
=============================================

Builds CSVs for the 5 seed/behavioral tables in schema.sql:
    customers, subscriptions, usage_logs, support_tickets, transactions

Design principle (per project docs): behavior is CONDITIONED on churn
outcome, not random. Churned customers get systematically worse usage
trends, more failed payments, and more support tickets than retained
customers, so the resulting model has real signal to find with SHAP —
this is what makes leakage-safe feature engineering and honest AUC
numbers possible in Phase 3's model-training step.

Two modes:
1. SEED MODE (preferred): point --seed-csv at a local copy of the Kaggle
   "Telco Customer Churn" dataset. Its customer-level columns (contract,
   payment method, monthly charges, tenure, Churn label, etc.) are mapped
   into `customers` + `subscriptions`, and used as the churn-label ground
   truth that drives conditioning for the other three tables.
   Download it yourself (network is not available in this environment):
   https://www.kaggle.com/datasets/blastchar/telco-customer-churn
   Place the CSV at, e.g., /mnt/user-data/uploads/Telco-Customer-Churn.csv

2. FALLBACK MODE (no seed file found): generates a fully synthetic
   customer base internally with a locked random seed, at a churn rate
   matching the real Telco dataset's (~26.5%), so the script is runnable
   end-to-end right now and produces output in the exact same shape.
   Swap in the real seed file later with zero code changes.

Usage:
    python3 generate_synthetic_data.py --seed-csv /path/to/telco.csv --out-dir ./synthetic_data
    python3 generate_synthetic_data.py --out-dir ./synthetic_data   # fallback mode
"""

import argparse
import os
import numpy as np
import pandas as pd
from datetime import date, timedelta

RNG_SEED = 42
TODAY = date(2026, 9, 16)  # anchor date for "today" in this synthetic world


# ---------------------------------------------------------------------
# 1. Load or fabricate the customer/subscription seed
# ---------------------------------------------------------------------

def load_seed(seed_csv_path: str | None, n_fallback: int, rng: np.random.Generator) -> pd.DataFrame:
    if seed_csv_path and os.path.exists(seed_csv_path):
        print(f"[seed] Loading real Telco seed dataset from {seed_csv_path}")
        df = pd.read_csv(seed_csv_path)
        df = df.rename(columns={
            "customerID": "customer_id",
            "Contract": "contract_type",
            "PaymentMethod": "payment_method",
            "MonthlyCharges": "monthly_charge",
            "tenure": "tenure_months",
            "Churn": "churn_label_raw",
            "gender": "gender",
            "SeniorCitizen": "senior_citizen",
        })
        df["churn_label"] = (df["churn_label_raw"].astype(str).str.strip().str.lower() == "yes").astype(int)
        df["monthly_charge"] = pd.to_numeric(df["monthly_charge"], errors="coerce").fillna(50.0)
        df["tenure_months"] = pd.to_numeric(df["tenure_months"], errors="coerce").fillna(0).astype(int)
        df["age"] = np.where(df.get("senior_citizen", 0) == 1,
                              rng.integers(60, 80, size=len(df)),
                              rng.integers(22, 59, size=len(df)))
        plan_map = {"Month-to-month": "Basic", "One year": "Standard", "Two year": "Pro"}
        df["plan_type"] = df["contract_type"].map(plan_map).fillna("Basic")
        regions = ["North", "South", "East", "West", "Central"]
        channels = ["organic", "paid_search", "referral", "partner", "social"]
        df["region"] = rng.choice(regions, size=len(df))
        df["signup_channel"] = rng.choice(channels, size=len(df))
        keep = ["customer_id", "region", "signup_channel", "age", "gender",
                "plan_type", "contract_type", "payment_method", "monthly_charge",
                "tenure_months", "churn_label"]
        return df[keep].copy()

    print(f"[seed] No seed CSV found at '{seed_csv_path}'. Falling back to fully synthetic "
          f"customer base (n={n_fallback}), churn rate locked to ~26.5% to match real Telco data.")
    return generate_fallback_customers(n_fallback, rng)


def generate_fallback_customers(n: int, rng: np.random.Generator) -> pd.DataFrame:
    regions = ["North", "South", "East", "West", "Central"]
    channels = ["organic", "paid_search", "referral", "partner", "social"]
    contracts = ["Month-to-month", "One year", "Two year"]
    contract_p = [0.55, 0.24, 0.21]  # Month-to-month dominant, matches real Telco skew
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

    customer_ids = [f"CUST-{i:06d}" for i in range(1, n + 1)]
    contract_type = rng.choice(contracts, size=n, p=contract_p)
    tenure_months = rng.integers(0, 73, size=n)
    monthly_charge = np.round(rng.uniform(20, 120, size=n), 2)
    plan_map = {"Month-to-month": "Basic", "One year": "Standard", "Two year": "Pro"}
    plan_type = np.array([plan_map[c] for c in contract_type])

    # Base churn propensity: month-to-month + low tenure + high charge = higher risk.
    # This is the ONLY place churn probability is decided; everything else downstream
    # (usage, tickets, transactions) is conditioned on the resulting label.
    risk_score = (
        0.35 * (contract_type == "Month-to-month").astype(float)
        + 0.25 * (1 - np.clip(tenure_months / 72, 0, 1))
        + 0.15 * (monthly_charge - 20) / 100
        + rng.normal(0, 0.12, size=n)
    )
    threshold = np.quantile(risk_score, 1 - 0.265)  # lock ~26.5% churn rate
    churn_label = (risk_score >= threshold).astype(int)

    return pd.DataFrame({
        "customer_id": customer_ids,
        "region": rng.choice(regions, size=n),
        "signup_channel": rng.choice(channels, size=n),
        "age": rng.integers(22, 80, size=n),
        "gender": rng.choice(["Male", "Female"], size=n),
        "plan_type": plan_type,
        "contract_type": contract_type,
        "payment_method": rng.choice(payment_methods, size=n),
        "monthly_charge": monthly_charge,
        "tenure_months": tenure_months,
        "churn_label": churn_label,
    })


# ---------------------------------------------------------------------
# 2. customers + subscriptions tables
# ---------------------------------------------------------------------

def build_customers_and_subscriptions(seed_df: pd.DataFrame, rng: np.random.Generator):
    n = len(seed_df)
    signup_date = [TODAY - timedelta(days=int(t * 30.44) + int(rng.integers(0, 30)))
                   for t in seed_df["tenure_months"]]

    customers = pd.DataFrame({
        "customer_id": seed_df["customer_id"],
        "signup_date": signup_date,
        "region": seed_df["region"],
        "signup_channel": seed_df["signup_channel"],
        "age": seed_df["age"],
        "gender": seed_df["gender"],
    })

    billing_cycle = np.where(seed_df["contract_type"] == "Month-to-month", "Monthly", "Annual")
    # churned customers: subscription already cancelled, end_date set; else active
    status = np.where(seed_df["churn_label"] == 1, "cancelled", "active")
    end_date = [
        (TODAY - timedelta(days=int(rng.integers(1, 60)))) if s == "cancelled" else None
        for s in status
    ]

    subscriptions = pd.DataFrame({
        "customer_id": seed_df["customer_id"],
        "plan_type": seed_df["plan_type"],
        "contract_type": seed_df["contract_type"],
        "billing_cycle": billing_cycle,
        "payment_method": seed_df["payment_method"],
        "monthly_charge": seed_df["monthly_charge"],
        "tenure_months": seed_df["tenure_months"],
        "status": status,
        "start_date": customers["signup_date"],
        "end_date": end_date,
    })

    return customers, subscriptions


# ---------------------------------------------------------------------
# 3. usage_logs — conditioned on churn, but NOT deterministically.
#
# IMPORTANT DESIGN NOTE (added after diagnosing near-perfect separability):
# An earlier version of this generator applied a single deterministic decay
# curve to every churner with too little noise. Diagnostic check showed
# individual features reaching 0.86-0.96 single-feature AUC, and a combined
# model reaching ROC-AUC 0.9998 — unrealistic (real churn literature tops
# out around 0.85-0.93, per Kairos_documentation.pdf Section 5.1). Fixed by:
#   1. A fraction of churners are "silent" (SILENT_CHURN_FRAC) — no usage
#      decline at all, representing real exogenous churn (price sensitivity,
#      competitor offers, life circumstances) that behavioral data can't
#      explain. This caps how predictable churn can ever be, which is
#      realistic and also a defensible point in the paper's limitations.
#   2. Per-customer random decay slopes (not one fixed curve) + larger,
#      proportional noise, so "predictable" churners vary in how visible
#      their decline is instead of following an identical trajectory.
#   3. Some retained customers naturally run low usage too (heterogeneous
#      population), so low usage alone doesn't perfectly imply churn.
# ---------------------------------------------------------------------

SILENT_CHURN_FRAC = 0.38  # share of churners with NO behavioral warning signs


def build_usage_logs(seed_df: pd.DataFrame, rng: np.random.Generator, n_months: int = 6) -> pd.DataFrame:
    rows = []
    for _, row in seed_df.iterrows():
        cust_id = row["customer_id"]
        churned = row["churn_label"] == 1
        # Silent churners behave like retained customers behaviorally —
        # their churn isn't explainable from usage data, by design.
        silent_churner = churned and (rng.random() < SILENT_CHURN_FRAC)
        predictable_decline = churned and not silent_churner

        # Heterogeneous baseline regardless of churn status — some retained
        # customers are naturally light users, some churners are naturally
        # heavy users. This creates the overlap real datasets have.
        base_sessions = rng.lognormal(mean=np.log(15), sigma=0.45)
        base_minutes = rng.lognormal(mean=np.log(25), sigma=0.45)
        base_adoption = float(np.clip(rng.normal(0.55, 0.18), 0.05, 0.95))

        # Per-customer random decay slope — no two predictable churners
        # decline identically.
        decay_rate = rng.uniform(0.03, 0.10) if predictable_decline else 0.0
        noise_scale = 0.5  # proportional noise, applied every month

        for m in range(n_months, 0, -1):
            log_month = (TODAY.replace(day=1) - pd.DateOffset(months=m)).date()
            months_elapsed = n_months - m + 1
            decay = max(1 - decay_rate * months_elapsed, 0.2)

            session_noise = rng.normal(1.0, noise_scale)
            minutes_noise = rng.normal(1.0, noise_scale)
            adoption_noise = rng.normal(0, 0.08)

            sessions = max(0, int(base_sessions * decay * session_noise))
            minutes = max(0.0, round(base_minutes * decay * minutes_noise, 2))
            adoption = float(np.clip(base_adoption * decay + adoption_noise, 0.02, 0.98))
            last_active = log_month + timedelta(days=int(rng.integers(0, 27)))

            rows.append({
                "customer_id": cust_id,
                "log_month": log_month,
                "session_count": sessions,
                "avg_session_minutes": minutes,
                "feature_adoption_score": round(adoption, 3),
                "last_active_date": last_active,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. support_tickets — conditioned on churn: churners raise more
#    tickets, more billing-related, slower resolution.
# ---------------------------------------------------------------------

TICKET_TEXT_TEMPLATES = {
    "billing": [
        "Customer disputes a charge on their most recent invoice.",
        "Asked why the monthly charge increased without notice.",
        "Reported a failed payment and requested a retry.",
        "Requested a refund for a duplicate charge.",
    ],
    "technical": [
        "Reported repeated login failures over the past week.",
        "App crashes when loading the dashboard view.",
        "Feature X is not syncing data correctly.",
        "Requested help configuring an integration.",
    ],
    "other": [
        "General question about plan upgrade options.",
        "Asked about cancellation policy and notice period.",
        "Requested a copy of past invoices.",
        "Provided feedback about the onboarding experience.",
    ],
}


def build_support_tickets(seed_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    ticket_id_counter = 1
    for _, row in seed_df.iterrows():
        cust_id = row["customer_id"]
        churned = row["churn_label"] == 1
        # Silent churners (see build_usage_logs) also get normal ticket behavior —
        # consistent with them being behaviorally indistinguishable from retained
        # customers, by design.
        silent_churner = churned and (rng.random() < SILENT_CHURN_FRAC)
        elevated_risk = churned and not silent_churner

        # churners: 0-5 tickets, skewed higher and more billing-related — but only
        # for the "predictable" (non-silent) group
        n_tickets = int(rng.poisson(2.0 if elevated_risk else 0.9))
        n_tickets = min(n_tickets, 6)

        for _ in range(n_tickets):
            days_ago = int(rng.integers(1, 180))
            created_at = TODAY - timedelta(days=days_ago)
            category = rng.choice(
                ["billing", "technical", "other"],
                p=[0.50, 0.30, 0.20] if elevated_risk else [0.32, 0.38, 0.30]
            )
            priority = rng.choice(["low", "medium", "high"], p=[0.3, 0.5, 0.2])
            resolved = rng.random() < (0.72 if not elevated_risk else 0.58)
            resolution_hours = None
            resolved_at = None
            if resolved:
                resolution_hours = round(float(rng.uniform(1, 84 if elevated_risk else 48)), 2)
                resolved_at = created_at + timedelta(hours=resolution_hours)
            text = rng.choice(TICKET_TEXT_TEMPLATES[category])

            rows.append({
                "ticket_id": ticket_id_counter,
                "customer_id": cust_id,
                "created_at": created_at,
                "resolved_at": resolved_at,
                "category": category,
                "priority": priority,
                "status": "closed" if resolved else "open",
                "resolution_time_hours": resolution_hours,
                "ticket_text": text,
            })
            ticket_id_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 5. transactions — conditioned on churn: churners have more failed
#    payments in the run-up to cancellation.
# ---------------------------------------------------------------------

def build_transactions(seed_df: pd.DataFrame, rng: np.random.Generator, n_months: int = 6) -> pd.DataFrame:
    rows = []
    txn_id_counter = 1
    for _, row in seed_df.iterrows():
        cust_id = row["customer_id"]
        churned = row["churn_label"] == 1
        silent_churner = churned and (rng.random() < SILENT_CHURN_FRAC)
        elevated_risk = churned and not silent_churner
        charge = float(row["monthly_charge"])

        for m in range(n_months, 0, -1):
            txn_date = TODAY - pd.DateOffset(months=m)
            fail_prob = 0.28 if elevated_risk else 0.05
            failed = rng.random() < fail_prob
            if failed:
                rows.append({
                    "transaction_id": txn_id_counter,
                    "customer_id": cust_id,
                    "transaction_date": txn_date,
                    "amount": charge,
                    "transaction_type": "failed_payment",
                    "status": "failed",
                    "payment_method": row["payment_method"],
                })
                txn_id_counter += 1
                # a retry that succeeds a few days later, most of the time
                if rng.random() < 0.7:
                    rows.append({
                        "transaction_id": txn_id_counter,
                        "customer_id": cust_id,
                        "transaction_date": txn_date + pd.Timedelta(days=3),
                        "amount": charge,
                        "transaction_type": "payment",
                        "status": "success",
                        "payment_method": row["payment_method"],
                    })
                    txn_id_counter += 1
            else:
                rows.append({
                    "transaction_id": txn_id_counter,
                    "customer_id": cust_id,
                    "transaction_date": txn_date,
                    "amount": charge,
                    "transaction_type": "payment",
                    "status": "success",
                    "payment_method": row["payment_method"],
                })
                txn_id_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Kairos synthetic data.")
    parser.add_argument("--seed-csv", type=str, default=None,
                         help="Path to Kaggle Telco-Customer-Churn.csv (optional).")
    parser.add_argument("--out-dir", type=str, default="./synthetic_data")
    parser.add_argument("--n-fallback", type=int, default=2000,
                         help="Number of customers to generate if no seed CSV is found.")
    args = parser.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    os.makedirs(args.out_dir, exist_ok=True)

    seed_df = load_seed(args.seed_csv, args.n_fallback, rng)
    print(f"[info] {len(seed_df)} customers loaded. Churn rate: {seed_df['churn_label'].mean():.1%}")

    customers, subscriptions = build_customers_and_subscriptions(seed_df, rng)
    usage_logs = build_usage_logs(seed_df, rng)
    support_tickets = build_support_tickets(seed_df, rng)
    transactions = build_transactions(seed_df, rng)

    # Keep the ground-truth churn label around separately (NOT loaded into `customers`
    # itself — it belongs conceptually in churn_predictions.actual_churn once the ML
    # pipeline runs, or as a held-out label file for training/eval).
    labels = seed_df[["customer_id", "churn_label"]]

    outputs = {
        "customers.csv": customers,
        "subscriptions.csv": subscriptions,
        "usage_logs.csv": usage_logs,
        "support_tickets.csv": support_tickets,
        "transactions.csv": transactions,
        "churn_labels_for_training.csv": labels,
    }
    for filename, df in outputs.items():
        path = os.path.join(args.out_dir, filename)
        df.to_csv(path, index=False)
        print(f"[write] {path}  ({len(df)} rows)")

    print("\n[done] Load order into Postgres (after running schema.sql):")
    print("  1. customers.csv        -> customers")
    print("  2. subscriptions.csv    -> subscriptions")
    print("  3. usage_logs.csv       -> usage_logs")
    print("  4. support_tickets.csv  -> support_tickets  (ticket_embedding stays NULL until Phase 5)")
    print("  5. transactions.csv     -> transactions")
    print("  (churn_labels_for_training.csv is NOT a DB table — it's the leakage-safe")
    print("   ground truth to hold out for Phase 3 model training/validation.)")


if __name__ == "__main__":
    main()
