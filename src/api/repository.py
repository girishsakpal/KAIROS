"""
Data access for the API.

CsvRepository reads the Phase 3 CSV/JSON outputs. It exists because Postgres has
not been loaded yet. Routes only talk to this class's methods, so a
PostgresRepository with the same methods can replace it later without touching
routes.py.

LEAKAGE / TIME RULE (same as build_features.py): every behavioural summary for a
customer uses only rows dated on or before THAT customer's snapshot_date, read
from features_train/test.csv. Customers with no snapshot fall back to
REFERENCE_DATE. So the numbers shown here match what the model saw.
"""
import datetime as dt
import decimal
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import summaries


def to_py(v):
    """numpy/pandas scalar -> plain JSON-safe Python (NaN/NaT -> None)."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if math.isnan(v) else float(v)
    if isinstance(v, decimal.Decimal):                  # Postgres NUMERIC
        return float(v)
    if isinstance(v, pd.Timestamp):                     # (checked before datetime: it is a subclass)
        return None if pd.isna(v) else v.date().isoformat()
    if isinstance(v, (dt.datetime, dt.date)):           # Postgres DATE / TIMESTAMPTZ
        return v.isoformat()
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def clean_feature_name(name: str) -> str:
    """'num__tenure_months' -> 'tenure_months'; 'cat__contract_type_One year' -> 'contract_type_One year'."""
    for prefix in ("num__", "cat__"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


class CsvRepository:
    def __init__(self, cfg):
        syn, proc, mod = Path(cfg["SYNTHETIC_DIR"]), Path(cfg["PROCESSED_DIR"]), Path(cfg["MODELS_DIR"])
        self.reference_date = pd.Timestamp(cfg["REFERENCE_DATE"])

        self.customers = pd.read_csv(syn / "customers.csv", parse_dates=["signup_date"]).set_index("customer_id")
        self.subs = pd.read_csv(syn / "subscriptions.csv").drop_duplicates("customer_id").set_index("customer_id")

        usage = pd.read_csv(syn / "usage_logs.csv", parse_dates=["log_month"])
        tickets = pd.read_csv(syn / "support_tickets.csv", parse_dates=["created_at"])
        txns = pd.read_csv(syn / "transactions.csv", parse_dates=["transaction_date"])
        self._usage = dict(tuple(usage.groupby("customer_id")))
        self._tickets = dict(tuple(tickets.groupby("customer_id")))
        self._txns = dict(tuple(txns.groupby("customer_id")))

        snaps = []
        for name in ("features_train.csv", "features_test.csv"):
            f = proc / name
            if f.exists():
                snaps.append(pd.read_csv(f, usecols=["customer_id", "snapshot_date"],
                                         parse_dates=["snapshot_date"]))
        self._snap = (pd.concat(snaps).drop_duplicates("customer_id").set_index("customer_id")["snapshot_date"]
                      if snaps else pd.Series(dtype="datetime64[ns]"))

        # Predictions + registry rows are optional: a missing file means 503, not a crash.
        pred_path = mod / cfg["PREDICTIONS_FILE"]
        self._pred = None
        if pred_path.exists():
            p = pd.read_csv(pred_path).sort_values("predicted_at")
            self._pred = p.drop_duplicates("customer_id", keep="last").set_index("customer_id")
        self._serving = self._read_json(mod / cfg["SERVING_REGISTRY_FILE"])
        self._champion = self._read_json(mod / cfg["CHAMPION_REGISTRY_FILE"])

    @staticmethod
    def _read_json(path):
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None

    # ---- lookups -------------------------------------------------------
    def customer_exists(self, cid):
        return cid in self.customers.index

    def snapshot_date(self, cid):
        if cid in self._snap.index and pd.notna(self._snap.loc[cid]):
            return self._snap.loc[cid]
        return self.reference_date

    def profile(self, cid):
        c, s = self.customers.loc[cid], self.subs.loc[cid] if cid in self.subs.index else None
        plan = f"{s['plan_type']} {s['billing_cycle']}" if s is not None else None
        return {"signup_date": to_py(c["signup_date"]), "region": to_py(c["region"]),
                "plan": plan, "signup_channel": to_py(c["signup_channel"]),
                "age": to_py(c["age"]), "gender": to_py(c["gender"])}

    def subscription(self, cid):
        if cid not in self.subs.index:
            return None
        s = self.subs.loc[cid]
        return {"status": to_py(s["status"]), "monthly_charge": to_py(s["monthly_charge"]),
                "tenure_months": to_py(s["tenure_months"]),
                "contract_type": to_py(s["contract_type"]), "payment_method": to_py(s["payment_method"])}

    def usage_summary(self, cid):
        return summaries.usage_summary(self._usage.get(cid), self.snapshot_date(cid))

    def support_summary(self, cid):
        return summaries.support_summary(self._tickets.get(cid), self.snapshot_date(cid))

    def transaction_summary(self, cid):
        return summaries.transaction_summary(self._txns.get(cid), self.snapshot_date(cid))

    # ---- predictions & models -----------------------------------------
    def scored_customer_ids(self):
        return [] if self._pred is None else list(self._pred.index)

    def latest_prediction(self, cid):
        if self._pred is None or cid not in self._pred.index:
            return None
        r = self._pred.loc[cid]
        raw = r.get("top_features")
        try:
            drivers = json.loads(raw) if isinstance(raw, str) else []
        except json.JSONDecodeError:
            drivers = []
        return {"churn_probability": round(float(r["churn_probability"]), 4),
                "risk_tier": to_py(r["risk_tier"]), "model_id": to_py(r["model_id"]),
                "predicted_at": to_py(r["predicted_at"]),
                "top_features": [{"feature": clean_feature_name(d["feature"]),
                                  "shap_value": d["shap_value"], "direction": d["direction"]}
                                 for d in drivers]}

    def serving_model(self):
        return self._serving

    def champion_model(self):
        return self._champion

    def model_by_id(self, model_id):
        for m in (self._champion, self._serving):
            if m and m.get("model_id") == model_id:
                return m
        return None

    def counts(self):
        return {"customers": int(len(self.customers)),
                "predictions": 0 if self._pred is None else int(len(self._pred))}
