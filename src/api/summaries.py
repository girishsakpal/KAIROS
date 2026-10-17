"""
Behavioural summaries shown by the API, as pure functions over a customer's rows.

BOTH repositories (CSV and Postgres) call these, so the numbers cannot drift apart: they differ only in
how rows are fetched. Each function takes that ONE customer's rows (a DataFrame or None) and the
customer's snapshot date, and ignores every row dated after the snapshot (the same leakage rule as
build_features.py).

Expected columns (datetime columns already tz-naive datetime64):
  usage:         log_month, session_count
  tickets:       created_at, status, category
  transactions:  transaction_date, transaction_type, status, amount
"""
import pandas as pd


def usage_summary(g, snap):
    """Usage logs are MONTHLY aggregates, so `sessions_30d` is the latest month on/before the snapshot.
    Trend compares the last 3 vs first 3 months (needs >= 4 months of data)."""
    if g is not None:
        g = g[g["log_month"] <= snap].sort_values("log_month")
    if g is None or g.empty:
        return {"sessions_30d": None, "sessions_90d_avg": None, "trend": "unknown"}
    recent, early = g.tail(3)["session_count"].mean(), g.head(3)["session_count"].mean()
    if len(g) < 4:
        trend = "unknown"
    else:
        rel = (recent - early) / early if early > 0 else 0.0
        trend = "declining" if rel < -0.10 else "growing" if rel > 0.10 else "stable"
    return {"sessions_30d": int(g.iloc[-1]["session_count"]),
            "sessions_90d_avg": round(float(recent), 1), "trend": trend}


def support_summary(g, snap):
    if g is not None:
        g = g[g["created_at"] <= snap]
    if g is None or g.empty:
        return {"open_tickets": 0, "tickets_60d": 0, "last_ticket_category": None}
    last = g.sort_values("created_at").iloc[-1]
    category = last["category"]
    return {"open_tickets": int((g["status"] == "open").sum()),
            "tickets_60d": int((g["created_at"] >= snap - pd.Timedelta(days=60)).sum()),
            "last_ticket_category": None if pd.isna(category) else category}


def transaction_summary(g, snap):
    if g is not None:
        g = g[g["transaction_date"] <= snap]
    if g is None or g.empty:
        return {"failed_payments_90d": 0, "lifetime_value": 0.0}
    failed = ((g["transaction_type"] == "failed_payment")
              & (g["transaction_date"] >= snap - pd.Timedelta(days=90))).sum()
    return {"failed_payments_90d": int(failed),
            "lifetime_value": round(float(g.loc[g["status"] == "success", "amount"].sum()), 2)}
