"""
Kairos — Feature Engineering (Phase 3)
=========================================

Builds a leakage-safe feature matrix from the synthetic (or real) tables in
data/synthetic/ (or data/raw/ once loaded into Postgres — this script reads
CSVs directly for now; swap read_csv calls for a DB query once the pipeline
moves off flat files).

LEAKAGE DISCIPLINE — read this before changing anything below
----------------------------------------------------------------
1. Fixed snapshot cutoff per customer.
   Every customer gets a `snapshot_date`: the point in time we pretend we're
   standing at when making the prediction.
     - Active customers -> snapshot_date = REFERENCE_DATE (today, in this
       synthetic world).
     - Churned customers -> snapshot_date = their subscription's end_date.
   ALL behavioral features (usage, tickets, transactions) are computed using
   ONLY rows dated on or before that customer's snapshot_date. Never later.
   This is what makes the eventual AUC/PR-AUC numbers honest instead of
   inflated by hindsight.

2. Outcome-proxy columns are excluded from the feature set entirely.
   `subscriptions.status` and `subscriptions.end_date` are not predictors —
   a cancelled subscription essentially *is* the churn label. They are used
   only to compute snapshot_date above, then dropped. Do not add them back
   as features; that would be leakage, not signal.

3. Time-based train/test split, not random.
   Customers are sorted by snapshot_date and split chronologically (earliest
   N% -> train, most recent -> test), simulating how the model would
   actually be deployed: trained on past snapshots, evaluated on more recent
   ones. A random split would let the model "see the future" via customers
   whose snapshot happens to fall in the same time window as a training
   customer's snapshot.

Output: data/processed/features_train.csv, features_test.csv,
        data/processed/feature_manifest.json (documents every column, for
        the SRS / report / viva — "how did you avoid leakage?" should be
        answerable by pointing at this file).
"""

import argparse
import json
import os
from datetime import date

import numpy as np
import pandas as pd

REFERENCE_DATE = pd.Timestamp(date(2026, 9, 16))  # must match generator's TODAY
TRAIN_FRACTION = 0.8


# ---------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------

def load_tables(data_dir: str):
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), parse_dates=["signup_date"])
    subscriptions = pd.read_csv(os.path.join(data_dir, "subscriptions.csv"),
                                 parse_dates=["start_date", "end_date"])
    usage_logs = pd.read_csv(os.path.join(data_dir, "usage_logs.csv"),
                              parse_dates=["log_month", "last_active_date"])
    support_tickets = pd.read_csv(os.path.join(data_dir, "support_tickets.csv"),
                                   parse_dates=["created_at", "resolved_at"])
    transactions = pd.read_csv(os.path.join(data_dir, "transactions.csv"),
                                parse_dates=["transaction_date"])
    labels = pd.read_csv(os.path.join(data_dir, "churn_labels_for_training.csv"))
    return customers, subscriptions, usage_logs, support_tickets, transactions, labels


# ---------------------------------------------------------------------
# 2. Snapshot cutoff per customer (the core leakage guard)
# ---------------------------------------------------------------------

def compute_snapshot_dates(subscriptions: pd.DataFrame, rng: np.random.Generator,
                            active_jitter_days: int = 120) -> pd.DataFrame:
    """
    Churned customers -> snapshot_date = their subscription end_date (already
    spread over the past ~60 days by the generator).

    Active customers -> WITHOUT jitter, every active customer would land on
    the exact same snapshot_date (today), collapsing a time-based split into
    a churn-vs-not split (verified: this happened on the first run). To
    simulate realistic periodic monitoring (scoring active customers on a
    rolling basis rather than all on one calendar day), each active customer
    is assigned a snapshot_date drawn uniformly from the last
    `active_jitter_days` days. This is a synthetic-data workaround — the
    underlying Telco-style dataset is cross-sectional by nature (no real
    time dimension), so a "real" time-based split will only make full sense
    once genuine rolling production data exists (Phase 6). Documented in
    the manifest below; revisit if the guide wants a different treatment.
    """
    snap = subscriptions[["customer_id", "status", "end_date"]].copy()
    n = len(snap)
    jitter_days = rng.integers(0, active_jitter_days, size=n)
    active_dates = REFERENCE_DATE - pd.to_timedelta(jitter_days, unit="D")

    snap["snapshot_date"] = np.where(
        snap["status"] == "cancelled",
        snap["end_date"],
        active_dates,
    )
    snap["snapshot_date"] = pd.to_datetime(snap["snapshot_date"])
    # status and end_date are dropped here on purpose — see module docstring.
    return snap[["customer_id", "snapshot_date"]]


# ---------------------------------------------------------------------
# 3. Behavioral features, each filtered to <= snapshot_date per customer
# ---------------------------------------------------------------------

def build_usage_features(usage_logs: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    merged = usage_logs.merge(snapshots, on="customer_id", how="inner")
    valid = merged[merged["log_month"] <= merged["snapshot_date"]].copy()
    valid = valid.sort_values(["customer_id", "log_month"])

    def agg(group):
        recent = group.tail(3)   # most recent up-to-3 months available before cutoff
        early = group.head(3)    # earliest up-to-3 months available before cutoff
        trend = recent["session_count"].mean() - early["session_count"].mean()
        last_active = group["last_active_date"].max()
        snap = group["snapshot_date"].iloc[0]
        return pd.Series({
            "avg_session_count_recent": recent["session_count"].mean(),
            "avg_session_minutes_recent": recent["avg_session_minutes"].mean(),
            "avg_feature_adoption_recent": recent["feature_adoption_score"].mean(),
            "session_count_trend": trend,
            "days_since_last_active": (snap - last_active).days if pd.notna(last_active) else np.nan,
            "months_of_usage_history": len(group),
        })

    feats = valid.groupby("customer_id").apply(agg, include_groups=False).reset_index()
    return feats


def build_ticket_features(support_tickets: pd.DataFrame, snapshots: pd.DataFrame,
                           window_days: int = 90) -> pd.DataFrame:
    merged = support_tickets.merge(snapshots, on="customer_id", how="inner")
    merged["created_at"] = pd.to_datetime(merged["created_at"])
    window_start = merged["snapshot_date"] - pd.Timedelta(days=window_days)
    valid = merged[(merged["created_at"] <= merged["snapshot_date"]) &
                   (merged["created_at"] >= window_start)].copy()

    def agg(group):
        n = len(group)
        billing_ratio = (group["category"] == "billing").mean() if n else 0.0
        open_count = (group["status"] == "open").sum()
        avg_resolution = group["resolution_time_hours"].mean()
        return pd.Series({
            "ticket_count_90d": n,
            "billing_ticket_ratio_90d": billing_ratio,
            "open_ticket_count": open_count,
            "avg_resolution_hours_90d": avg_resolution,
        })

    if valid.empty:
        return pd.DataFrame(columns=["customer_id", "ticket_count_90d",
                                      "billing_ticket_ratio_90d", "open_ticket_count",
                                      "avg_resolution_hours_90d"])
    feats = valid.groupby("customer_id").apply(agg, include_groups=False).reset_index()
    return feats


def build_transaction_features(transactions: pd.DataFrame, snapshots: pd.DataFrame,
                                window_days: int = 90) -> pd.DataFrame:
    merged = transactions.merge(snapshots, on="customer_id", how="inner")
    merged["transaction_date"] = pd.to_datetime(merged["transaction_date"])
    window_start = merged["snapshot_date"] - pd.Timedelta(days=window_days)
    valid = merged[(merged["transaction_date"] <= merged["snapshot_date"]) &
                   (merged["transaction_date"] >= window_start)].copy()

    def agg(group):
        n = len(group)
        failed = (group["transaction_type"] == "failed_payment").sum()
        total_paid = group.loc[group["status"] == "success", "amount"].sum()
        return pd.Series({
            "failed_payment_count_90d": failed,
            "failed_payment_ratio_90d": failed / n if n else 0.0,
            "total_paid_90d": total_paid,
            "transaction_count_90d": n,
        })

    if valid.empty:
        return pd.DataFrame(columns=["customer_id", "failed_payment_count_90d",
                                      "failed_payment_ratio_90d", "total_paid_90d",
                                      "transaction_count_90d"])
    feats = valid.groupby("customer_id").apply(agg, include_groups=False).reset_index()
    return feats


# ---------------------------------------------------------------------
# 4. Static features (safe — not outcome proxies)
# ---------------------------------------------------------------------

def build_static_features(customers: pd.DataFrame, subscriptions: pd.DataFrame) -> pd.DataFrame:
    # Deliberately excludes subscriptions.status and subscriptions.end_date.
    sub_static = subscriptions[["customer_id", "plan_type", "contract_type",
                                 "billing_cycle", "payment_method", "monthly_charge",
                                 "tenure_months"]]
    static = customers[["customer_id", "region", "signup_channel", "age", "gender"]].merge(
        sub_static, on="customer_id", how="left"
    )
    return static


# ---------------------------------------------------------------------
# 5. Assemble + time-based split
# ---------------------------------------------------------------------

def assemble_features(data_dir: str, rng: np.random.Generator) -> pd.DataFrame:
    customers, subscriptions, usage_logs, support_tickets, transactions, labels = load_tables(data_dir)

    snapshots = compute_snapshot_dates(subscriptions, rng)
    static = build_static_features(customers, subscriptions)
    usage_feats = build_usage_features(usage_logs, snapshots)
    ticket_feats = build_ticket_features(support_tickets, snapshots)
    txn_feats = build_transaction_features(transactions, snapshots)

    df = snapshots.merge(static, on="customer_id", how="left")
    df = df.merge(usage_feats, on="customer_id", how="left")
    df = df.merge(ticket_feats, on="customer_id", how="left")
    df = df.merge(txn_feats, on="customer_id", how="left")
    df = df.merge(labels, on="customer_id", how="left")

    # Customers with no tickets/failed payments in window legitimately have zero,
    # not missing — fill those specific count/ratio columns with 0.
    zero_fill_cols = ["ticket_count_90d", "billing_ticket_ratio_90d", "open_ticket_count",
                       "failed_payment_count_90d", "failed_payment_ratio_90d",
                       "total_paid_90d", "transaction_count_90d"]
    for col in zero_fill_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df


def time_based_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION):
    df_sorted = df.sort_values("snapshot_date").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * train_fraction)
    cutoff_date = df_sorted.iloc[cutoff_idx]["snapshot_date"]
    train = df_sorted[df_sorted["snapshot_date"] < cutoff_date]
    test = df_sorted[df_sorted["snapshot_date"] >= cutoff_date]
    return train, test, cutoff_date


# ---------------------------------------------------------------------
# 6. Manifest — documents every column for the report/viva
# ---------------------------------------------------------------------

def build_manifest(df: pd.DataFrame, cutoff_date, train_n: int, test_n: int) -> dict:
    excluded = ["subscriptions.status", "subscriptions.end_date"]
    return {
        "reference_date": str(REFERENCE_DATE.date()),
        "snapshot_logic": (
            "Active customers: snapshot_date = reference_date minus a random jitter "
            "of 0-120 days (simulates rolling periodic monitoring; without this, all "
            "active customers collapse onto one calendar date and break the time-based "
            "split — verified during development). "
            "Cancelled customers: snapshot_date = subscription end_date. "
            "All behavioral features use only rows dated on/before snapshot_date."
        ),
        "known_limitation": (
            "The underlying Telco-style dataset is cross-sectional (no real time "
            "dimension), so the active-customer jitter is a synthetic-data workaround, "
            "not genuine temporal data. Revisit once real rolling production data "
            "exists (Phase 6)."
        ),
        "excluded_outcome_proxy_columns": excluded,
        "excluded_reason": (
            "status and end_date are direct proxies for the churn label itself "
            "and were used only to compute snapshot_date, never as model inputs."
        ),
        "split_method": "time-based (chronological by snapshot_date), not random",
        "train_test_cutoff_date": str(pd.Timestamp(cutoff_date).date()),
        "train_rows": train_n,
        "test_rows": test_n,
        "feature_columns": sorted([c for c in df.columns
                                    if c not in ("customer_id", "snapshot_date", "churn_label")]),
        "label_column": "churn_label",
        "windows": {
            "support_tickets": "trailing 90 days before snapshot_date",
            "transactions": "trailing 90 days before snapshot_date",
            "usage_logs": "all available months on/before snapshot_date; "
                          "recent/early split uses up to 3 months each"
        },
    }


# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kairos leakage-safe feature engineering.")
    parser.add_argument("--data-dir", type=str, default="data/synthetic")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--train-fraction", type=float, default=TRAIN_FRACTION)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    print(f"[info] Building features from {args.data_dir} ...")
    df = assemble_features(args.data_dir, rng)
    print(f"[info] {len(df)} customers, {df.shape[1]} columns before split.")
    print(f"[info] Overall churn rate: {df['churn_label'].mean():.1%}")

    train, test, cutoff_date = time_based_split(df, args.train_fraction)
    print(f"[info] Time-based split at snapshot_date >= {cutoff_date.date()}")
    print(f"[info] Train: {len(train)} rows (churn rate {train['churn_label'].mean():.1%})")
    print(f"[info] Test:  {len(test)} rows (churn rate {test['churn_label'].mean():.1%})")

    if train["churn_label"].nunique() < 2 or test["churn_label"].nunique() < 2:
        print("[warn] One of the splits has only a single class present. "
              "The time-based split may still be too degenerate for this dataset's "
              "temporal spread — inspect feature_manifest.json and consider adjusting "
              "active_jitter_days or train_fraction.")

    train_path = os.path.join(args.out_dir, "features_train.csv")
    test_path = os.path.join(args.out_dir, "features_test.csv")
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"[write] {train_path}")
    print(f"[write] {test_path}")

    manifest = build_manifest(df, cutoff_date, len(train), len(test))
    manifest_path = os.path.join(args.out_dir, "feature_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
