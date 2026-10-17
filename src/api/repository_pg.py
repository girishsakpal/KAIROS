"""
PostgresRepository: same public methods as CsvRepository, backed by the Supabase/Postgres tables.

Design choices
  * Queries are deliberately simple (`SELECT ... WHERE customer_id = %s`). All calculations happen in
    summaries.py, shared with CsvRepository, so the two cannot disagree on logic. The parity check
    (python -m src.api.check_parity) compares them over every scored customer.
  * The serving model is DERIVED from the data, not hard-coded: the most recently trained model that
    has per-customer SHAP drivers (the LSTM champion has none). Retrain, reload, and the API follows
    (after the short cache below expires or on restart).
  * Timestamps: tz-aware values are converted to UTC then made naive, so results do not depend on the
    session time zone. The loader stores date-only values as midnight UTC.
  * Connections come from a small pool; a dropped connection (Supabase closes idle ones) is retried once.
"""
import json
import time

import pandas as pd

from . import summaries
from .repository import clean_feature_name, to_py

MODEL_COLS = ("model_id, model_name, algorithm, trained_at, roc_auc, pr_auc, calibration_error, "
              "drift_psi_score, drift_ks_pvalue, drift_detected, status")
_CONNECTION_ERRORS = {"OperationalError", "InterfaceError"}


def _naive_utc(series):
    return pd.to_datetime(series, utc=True).dt.tz_localize(None)


def _num(series):
    return series.map(lambda v: None if v is None else float(v)).astype(float)


class PostgresRepository:
    def __init__(self, acquire, release, reference_date, model_cache_seconds=30):
        """acquire() -> DB-API connection; release(conn, broken=False). Injected so tests can use SQLite."""
        self._acquire, self._release = acquire, release
        self.reference_date = pd.Timestamp(reference_date)
        self._ttl = model_cache_seconds
        self._model_cache = {}
        self._snap_cache = {}

    @classmethod
    def from_env(cls, reference_date, minconn=1, maxconn=5):
        import os
        from dotenv import load_dotenv
        from psycopg2.pool import ThreadedConnectionPool
        from ..db.load_to_postgres import ROOT, masked, normalize_url
        load_dotenv(ROOT / ".env")
        url = normalize_url(os.getenv("DATABASE_URL", ""))
        pool = ThreadedConnectionPool(minconn, maxconn, dsn=url, connect_timeout=15)
        repo = cls(pool.getconn, lambda conn, broken=False: pool.putconn(conn, close=broken), reference_date)
        repo.description = masked(url)
        return repo

    # ---- plumbing ------------------------------------------------------
    def _rows(self, sql, params=()):
        """Run a read-only query; return list[dict]. Retries once on a dropped connection."""
        for attempt in (1, 2):
            conn = self._acquire()
            broken = False
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                out = [dict(zip(cols, r)) for r in cur.fetchall()]
                conn.rollback()                                  # close the read-only transaction
                return out
            except Exception as e:                               # noqa: BLE001
                broken = type(e).__name__ in _CONNECTION_ERRORS
                if not broken or attempt == 2:
                    raise
            finally:
                self._release(conn, broken)

    def _frame(self, sql, cid, dates=(), nums=()):
        rows = self._rows(sql, (cid,))
        if not rows:
            return None
        df = pd.DataFrame(rows)
        for c in dates:
            df[c] = _naive_utc(df[c])
        for c in nums:
            df[c] = _num(df[c])
        return df

    # ---- lookups -------------------------------------------------------
    def customer_exists(self, cid):
        return bool(self._rows("SELECT 1 AS one FROM customers WHERE customer_id = %s", (cid,)))

    def snapshot_date(self, cid):
        if cid not in self._snap_cache:
            rows = self._rows("SELECT snapshot_date FROM feature_snapshots WHERE customer_id = %s", (cid,))
            val = rows[0]["snapshot_date"] if rows else None
            self._snap_cache[cid] = pd.Timestamp(val) if val is not None else self.reference_date
        return self._snap_cache[cid]

    def _latest_subscription(self, cid, cols):
        rows = self._rows(f"SELECT {cols} FROM subscriptions WHERE customer_id = %s "
                          "ORDER BY start_date DESC LIMIT 1", (cid,))
        return rows[0] if rows else None

    def profile(self, cid):
        c = self._rows("SELECT signup_date, region, signup_channel, age, gender FROM customers "
                       "WHERE customer_id = %s", (cid,))[0]
        s = self._latest_subscription(cid, "plan_type, billing_cycle")
        return {"signup_date": to_py(c["signup_date"]), "region": to_py(c["region"]),
                "plan": f"{s['plan_type']} {s['billing_cycle']}" if s else None,
                "signup_channel": to_py(c["signup_channel"]), "age": to_py(c["age"]),
                "gender": to_py(c["gender"])}

    def subscription(self, cid):
        s = self._latest_subscription(cid, "status, monthly_charge, tenure_months, contract_type, payment_method")
        if s is None:
            return None
        return {"status": to_py(s["status"]), "monthly_charge": to_py(s["monthly_charge"]),
                "tenure_months": to_py(s["tenure_months"]), "contract_type": to_py(s["contract_type"]),
                "payment_method": to_py(s["payment_method"])}

    def usage_summary(self, cid):
        g = self._frame("SELECT log_month, session_count FROM usage_logs WHERE customer_id = %s", cid,
                        dates=("log_month",), nums=("session_count",))
        return summaries.usage_summary(g, self.snapshot_date(cid))

    def support_summary(self, cid):
        g = self._frame("SELECT created_at, status, category FROM support_tickets WHERE customer_id = %s",
                        cid, dates=("created_at",))
        return summaries.support_summary(g, self.snapshot_date(cid))

    def transaction_summary(self, cid):
        g = self._frame("SELECT transaction_date, transaction_type, status, amount FROM transactions "
                        "WHERE customer_id = %s", cid, dates=("transaction_date",), nums=("amount",))
        return summaries.transaction_summary(g, self.snapshot_date(cid))

    # ---- models & predictions -----------------------------------------
    @staticmethod
    def _model_dict(row):
        if row is None:
            return None
        out = {k: to_py(v) for k, v in row.items()}
        out["drift_detected"] = bool(out["drift_detected"])
        return out

    def _cached(self, key, loader):
        hit = self._model_cache.get(key)
        if hit and self._ttl and time.monotonic() - hit[0] < self._ttl:
            return hit[1]
        value = loader()
        self._model_cache[key] = (time.monotonic(), value)
        return value

    def serving_model(self):
        def load():
            rows = self._rows(f"SELECT {MODEL_COLS} FROM model_registry m WHERE EXISTS "
                              "(SELECT 1 FROM churn_predictions p WHERE p.model_id = m.model_id "
                              "AND jsonb_array_length(p.top_features) > 0) "
                              "ORDER BY m.trained_at DESC LIMIT 1")
            return self._model_dict(rows[0] if rows else None)
        return self._cached("serving", load)

    def champion_model(self):
        def load():
            rows = self._rows(f"SELECT {MODEL_COLS} FROM model_registry WHERE status = 'champion' "
                              "ORDER BY trained_at DESC LIMIT 1")
            return self._model_dict(rows[0] if rows else None)
        return self._cached("champion", load)

    def model_by_id(self, model_id):
        rows = self._rows(f"SELECT {MODEL_COLS} FROM model_registry WHERE model_id = %s", (model_id,))
        return self._model_dict(rows[0] if rows else None)

    def latest_prediction(self, cid):
        model = self.serving_model()
        if model is None:
            return None
        rows = self._rows("SELECT churn_probability, risk_tier, model_id, predicted_at, top_features "
                          "FROM churn_predictions WHERE customer_id = %s AND model_id = %s "
                          "ORDER BY predicted_at DESC LIMIT 1", (cid, model["model_id"]))
        if not rows:
            return None
        r = rows[0]
        drivers = r["top_features"]
        if isinstance(drivers, str):                    # SQLite / text; psycopg2 already returns a list for JSONB
            drivers = json.loads(drivers)
        return {"churn_probability": round(float(r["churn_probability"]), 4),
                "risk_tier": to_py(r["risk_tier"]), "model_id": to_py(r["model_id"]),
                "predicted_at": to_py(r["predicted_at"]),
                "top_features": [{"feature": clean_feature_name(d["feature"]),
                                  "shap_value": d["shap_value"], "direction": d["direction"]}
                                 for d in drivers]}

    def counts(self):
        model = self.serving_model()
        n_pred = 0
        if model:
            n_pred = self._rows("SELECT count(*) AS n FROM churn_predictions WHERE model_id = %s",
                                (model["model_id"],))[0]["n"]
        return {"customers": int(self._rows("SELECT count(*) AS n FROM customers")[0]["n"]),
                "predictions": int(n_pred)}
