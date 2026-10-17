"""
Tests for src/db/load_to_postgres.py.

These DO NOT talk to a real Postgres. They test (a) validation, (b) the COPY payloads, and
(c) the exact sequence of SQL the loader issues, using a fake connection. The first run against a real
Supabase project is the real integration test; use --dry-run first and --verify-only after.

    python -m unittest tests.test_db_loader -v
"""
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.db import load_to_postgres as L  # noqa: E402

SCHEMA_SQL = (ROOT / "db" / "schema.sql").read_text()
SCHEMA = L.parse_schema(SCHEMA_SQL)
TOP = json.dumps([{"feature": "num__tenure_months", "shap_value": -0.2, "direction": "decreases_risk"},
                  {"feature": "cat__x", "shap_value": 0.1, "direction": "increases_risk"}])


def write_inputs(root: Path, lstm_blank=True):
    syn, mod, proc = root / "syn", root / "models", root / "proc"
    syn.mkdir()
    mod.mkdir()
    proc.mkdir()
    pd.DataFrame({"customer_id": ["C1"], "snapshot_date": ["2026-09-01"],
                  "tenure_months": [19]}).to_csv(proc / "features_train.csv", index=False)
    pd.DataFrame({"customer_id": ["C2"], "snapshot_date": ["2026-08-29 00:00:00"],
                  "tenure_months": [5]}).to_csv(proc / "features_test.csv", index=False)
    pd.DataFrame({"customer_id": ["C1", "C2"], "signup_date": ["2025-01-01", "2024-06-01"],
                  "region": ["West", "East"], "signup_channel": ["organic", None], "age": [30, 41],
                  "gender": ["Male", "Female"]}).to_csv(syn / "customers.csv", index=False)
    pd.DataFrame({"customer_id": ["C1", "C2"], "plan_type": ["Pro", "Basic"],
                  "contract_type": ["Two year", "Month-to-month"], "billing_cycle": ["Annual", "Monthly"],
                  "payment_method": ["Credit card", "Mailed check"], "monthly_charge": [49.99, 20.5],
                  "tenure_months": [19, 5], "status": ["active", "cancelled"],
                  "start_date": ["2025-01-01", "2024-06-01"],
                  "end_date": [None, "2026-08-01"]}).to_csv(syn / "subscriptions.csv", index=False)
    pd.DataFrame({"customer_id": ["C1", "C1"], "log_month": ["2026-07-01", "2026-08-01"],
                  "session_count": [10, 8], "avg_session_minutes": [30.5, 28.0],
                  "feature_adoption_score": [0.5, 0.4],
                  "last_active_date": ["2026-07-30", None]}).to_csv(syn / "usage_logs.csv", index=False)
    pd.DataFrame({"ticket_id": [1, 2], "customer_id": ["C1", "C2"],
                  "created_at": ["2026-08-01", "2026-08-02"], "resolved_at": ["2026-08-03", None],
                  "category": ["billing", "other"], "priority": ["low", "high"], "status": ["closed", "open"],
                  "resolution_time_hours": [48.0, None],
                  "ticket_text": ['Charged twice, "urgent", please help', "Other"]}).to_csv(syn / "support_tickets.csv", index=False)
    pd.DataFrame({"transaction_id": [1, 2], "customer_id": ["C1", "C2"],
                  "transaction_date": ["2026-08-01", "2026-08-02"], "amount": [50.0, 20.5],
                  "transaction_type": ["payment", "failed_payment"], "status": ["success", "failed"],
                  "payment_method": ["Credit card", "Mailed check"]}).to_csv(syn / "transactions.csv", index=False)

    base = {"algorithm": "XGBoost", "trained_at": "2026-10-03T00:00:00+00:00", "feature_list": ["a", "b"],
            "roc_auc": 0.8936, "pr_auc": 0.85, "calibration_error": 0.13, "drift_psi_score": None,
            "drift_ks_pvalue": None, "drift_detected": False}
    (mod / "xgboost_registry_row.json").write_text(json.dumps(
        {**base, "model_id": "xgb_churn_v2", "model_name": "XGB", "status": "challenger"}))
    (mod / "model_registry_row.json").write_text(json.dumps(
        {**base, "model_id": "lstm_churn_v1", "model_name": "LSTM", "algorithm": "LSTM", "status": "champion"}))

    pd.DataFrame({"customer_id": ["C1", "C2"], "model_id": "xgb_churn_v2",
                  "predicted_at": "2026-10-05T01:00:00+00:00", "churn_probability": [0.91234567, 0.1],
                  "risk_tier": ["high", "low"], "actual_churn": [1, 0], "top_features": [TOP, TOP],
                  "churn_probability_raw": [0.99, 0.05]}).to_csv(mod / "xgboost_predictions_test.csv", index=False)
    pd.DataFrame({"customer_id": ["C1", "C2"], "model_id": "lstm_churn_v1",
                  "predicted_at": "2026-09-27T07:00:00+00:00", "churn_probability": [0.8, 0.2],
                  "risk_tier": ["high", "low"], "actual_churn": [1, 0],
                  "top_features": [None, None] if lstm_blank else [TOP, TOP]}).to_csv(mod / "churn_predictions_test.csv", index=False)
    return syn, mod, proc


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.syn, self.mod, self.proc = write_inputs(Path(self._tmp.name))
        self.frames, self.warnings = L.load_frames(self.syn, self.mod, processed_dir=self.proc)


class TestSchemaParsing(unittest.TestCase):
    def test_parses_all_tables_and_limits(self):
        self.assertEqual(set(SCHEMA), set(L.TABLES))
        self.assertEqual(SCHEMA["customers"]["customer_id"], {"maxlen": 20, "notnull": True})
        self.assertTrue(SCHEMA["churn_predictions"]["top_features"]["notnull"])
        self.assertFalse(SCHEMA["subscriptions"]["end_date"]["notnull"])
        self.assertEqual(SCHEMA["model_registry"]["model_name"]["maxlen"], 100)

    def test_every_copy_column_exists_in_schema(self):
        for t in L.TABLES:
            for c in L.COPY_COLUMNS[t]:
                self.assertIn(c, SCHEMA[t], f"{t}.{c}")


class TestLoadFrames(Base):
    def test_preflight_passes_on_clean_inputs(self):
        self.assertEqual(L.preflight(self.frames, SCHEMA), [])

    def test_all_blank_champion_drivers_become_empty_array(self):
        p = self.frames["churn_predictions"]
        lstm = p[p.model_id == "lstm_churn_v1"]
        self.assertEqual(set(lstm.top_features), {"[]"})
        self.assertTrue(any("no per-customer drivers" in w for w in self.warnings))
        xgb = p[p.model_id == "xgb_churn_v2"]
        self.assertTrue(all(len(json.loads(x)) == 2 for x in xgb.top_features))

    def test_booleans_and_extra_columns(self):
        p = self.frames["churn_predictions"]
        self.assertEqual(set(p.actual_churn), {"true", "false"})
        self.assertEqual(self.frames["model_registry"].drift_detected.tolist(), ["false", "false"])
        self.assertEqual(len(self.frames["model_registry"]), 2)

    def test_skip_champion_predictions(self):
        frames, _ = L.load_frames(self.syn, self.mod, include_champion=False, processed_dir=self.proc)
        self.assertEqual(set(frames["churn_predictions"].model_id), {"xgb_churn_v2"})

    def test_duplicate_prediction_rows_dropped(self):
        pd.read_csv(self.mod / "xgboost_predictions_test.csv").to_csv(self.mod / "churn_predictions_test.csv", index=False)
        frames, warns = L.load_frames(self.syn, self.mod, processed_dir=self.proc)
        self.assertEqual(len(frames["churn_predictions"]), 2)
        self.assertTrue(any("duplicate" in w for w in warns))


class TestSnapshots(Base):
    def test_snapshot_frame_has_split_and_clean_dates(self):
        f = self.frames["feature_snapshots"].set_index("customer_id")
        self.assertEqual(f.loc["C1", "split"], "train")
        self.assertEqual(f.loc["C2", "split"], "test")
        self.assertEqual(f.loc["C2", "snapshot_date"], "2026-08-29")      # time part stripped
        self.assertEqual(list(self.frames["feature_snapshots"].columns), L.COPY_COLUMNS["feature_snapshots"])

    def test_missing_processed_dir_warns_and_fails_preflight(self):
        frames, warns = L.load_frames(self.syn, self.mod)
        self.assertTrue(any("feature_snapshots will be empty" in w for w in warns))
        self.assertTrue(any("feature_snapshots is empty" in e for e in L.preflight(frames, SCHEMA)))

    def test_snapshot_for_unknown_customer(self):
        self.frames["feature_snapshots"].loc[0, "customer_id"] = "GHOST"
        self.assertTrue(any("feature_snapshots" in e and "not in customers" in e
                            for e in L.preflight(self.frames, SCHEMA)))

    def test_customer_in_both_splits(self):
        f = self.frames["feature_snapshots"]
        self.frames["feature_snapshots"] = pd.concat([f, f.iloc[[0]].assign(split="test")], ignore_index=True)
        self.assertTrue(any("both train and test" in e for e in L.preflight(self.frames, SCHEMA)))

    def test_bad_split_value(self):
        self.frames["feature_snapshots"].loc[0, "split"] = "validation"
        self.assertTrue(any("unexpected split" in e for e in L.preflight(self.frames, SCHEMA)))


class TestPreflightCatchesProblems(Base):
    def errs(self):
        return L.preflight(self.frames, SCHEMA)

    def test_orphan_foreign_key(self):
        self.frames["transactions"].loc[0, "customer_id"] = "GHOST"
        self.assertTrue(any("transactions" in e and "not in customers" in e for e in self.errs()))

    def test_duplicate_usage_month(self):
        u = self.frames["usage_logs"]
        self.frames["usage_logs"] = pd.concat([u, u.iloc[[0]]], ignore_index=True)
        self.assertTrue(any("usage_logs" in e and "UNIQUE" in e for e in self.errs()))

    def test_varchar_too_long(self):
        self.frames["customers"].loc[0, "customer_id"] = "X" * 21
        self.assertTrue(any("exceeds VARCHAR(20)" in e for e in self.errs()))

    def test_null_in_not_null_column(self):
        self.frames["customers"].loc[0, "region"] = None
        self.assertTrue(any("customers.region" in e and "NOT NULL" in e for e in self.errs()))

    def test_bad_json_in_top_features(self):
        self.frames["churn_predictions"].loc[0, "top_features"] = "{not json"
        self.assertTrue(any("not a JSON array" in e for e in self.errs()))

    def test_prediction_for_unregistered_model(self):
        self.frames["churn_predictions"].loc[0, "model_id"] = "mystery_v9"
        self.assertTrue(any("no model_registry row" in e for e in self.errs()))

    def test_probability_out_of_range(self):
        self.frames["churn_predictions"].loc[0, "churn_probability"] = 1.5
        self.assertTrue(any("outside [0, 1]" in e for e in self.errs()))

    def test_two_champions(self):
        self.frames["model_registry"]["status"] = "champion"
        self.assertTrue(any("more than one" in e for e in self.errs()))


class TestCopyBuffer(Base):
    def parse(self, buf):
        return list(csv.reader(buf))

    def test_null_quotes_and_json_round_trip(self):
        t = "support_tickets"
        rows = self.parse(L.to_copy_buffer(self.frames[t], L.COPY_COLUMNS[t]))
        self.assertEqual(rows[0][0], "1")
        self.assertEqual(rows[0][8], 'Charged twice, "urgent", please help')   # commas + quotes survive
        self.assertEqual(rows[1][3], "")                                       # NULL resolved_at -> empty field
        self.assertEqual(rows[1][7], "")

    def test_prediction_json_and_bool_columns(self):
        t = "churn_predictions"
        cols = L.COPY_COLUMNS[t]
        rows = self.parse(L.to_copy_buffer(self.frames[t], cols))
        self.assertEqual(json.loads(rows[0][cols.index("top_features")])[0]["feature"], "num__tenure_months")
        self.assertIn(rows[0][cols.index("actual_churn")], {"true", "false"})
        self.assertEqual(len(rows[0]), len(cols))          # churn_probability_raw is NOT emitted

    def test_ids_are_integers_not_floats(self):
        rows = self.parse(L.to_copy_buffer(self.frames["transactions"], L.COPY_COLUMNS["transactions"]))
        self.assertEqual([r[0] for r in rows], ["1", "2"])


# ---------------------------------------------------------------- fake database
class FakeCursor:
    def __init__(self, conn):
        self.conn, self._rows = conn, []

    def execute(self, sql, params=None):
        self.conn.events.append(("sql", " ".join(sql.split())))
        for key in self.conn.fail_on:
            if key in sql:
                raise RuntimeError(f"boom: {key}")
        flat = " ".join(sql.split())
        for sub, rows in self.conn.script:
            if sub in flat:
                self._rows = rows() if callable(rows) else rows
                return
        for t in L.TABLES:
            if flat == f"SELECT count(*) FROM {t}":
                self._rows = [(self.conn.counts.get(t, 0),)]
                return
        self._rows = []

    def copy_expert(self, sql, buf):
        self.conn.events.append(("copy", " ".join(sql.split())))
        for key in self.conn.fail_on:
            if key in sql:
                raise RuntimeError(f"boom: {key}")

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def fetchall(self):
        return self._rows


class FakeConn:
    def __init__(self, tables_present=None, counts=None, fail_on=(), script=()):
        self.events, self.autocommit = [], False
        self.tables_present = list(tables_present or [])
        self.counts, self.fail_on = counts or {}, list(fail_on)
        # test-supplied answers take priority over the defaults (first match wins)
        self.script = list(script) + [("FROM pg_extension", [(1,)]),
                                      ("information_schema.tables", lambda: [(t,) for t in self.tables_present])]

    def cursor(self):
        return FakeCursor(self)

    def commit(self):
        self.events.append(("commit", ""))

    def rollback(self):
        self.events.append(("rollback", ""))

    def close(self):
        pass

    def sqls(self):
        return [e[1] for e in self.events if e[0] == "sql"]

    def kinds(self):
        return [e[0] for e in self.events]


QUIET = lambda *a, **k: None   # noqa: E731


class TestRunLoad(Base):
    def test_fresh_database_creates_schema_loads_in_order_and_commits_once(self):
        conn = FakeConn()
        L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        copies = [e[1].split()[1] for e in conn.events if e[0] == "copy"]
        # hard-coded on purpose: parents must load before children (FK-safe)
        self.assertEqual(copies, ["customers", "subscriptions", "usage_logs", "support_tickets",
                                  "transactions", "feature_snapshots", "model_registry", "churn_predictions"])
        sqls = conn.sqls()
        self.assertTrue(any(s.startswith("SET search_path TO public, extensions") for s in sqls))
        self.assertTrue(any("CREATE TABLE customers" in s for s in sqls))   # schema applied
        self.assertEqual(sum(1 for s in sqls if "ENABLE ROW LEVEL SECURITY" in s), 8)

        for t in L.EXPLICIT_ID_TABLES:
            self.assertTrue(any("setval" in s and t in s for s in sqls))
        self.assertEqual(conn.kinds().count("commit"), 1)
        self.assertEqual(conn.kinds()[-1], "commit")
        self.assertNotIn("rollback", conn.kinds())
        # RLS and sequences happen after all data is loaded, commit last
        last_copy = max(i for i, e in enumerate(conn.events) if e[0] == "copy")
        first_rls = min(i for i, e in enumerate(conn.events) if "ROW LEVEL SECURITY" in e[1])
        self.assertLess(last_copy, first_rls)

    def test_refuses_non_empty_database_without_reset(self):
        conn = FakeConn(tables_present=L.TABLES, counts={"customers": 5})
        with self.assertRaises(SystemExit) as cm:
            L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        self.assertIn("--reset", str(cm.exception))
        self.assertNotIn("commit", conn.kinds())
        self.assertIn("rollback", conn.kinds())
        self.assertFalse(any(e[0] == "copy" for e in conn.events))

    def test_reset_truncates_children_first_then_loads(self):
        conn = FakeConn(tables_present=L.TABLES, counts={"customers": 5})
        L.run_load(conn, self.frames, SCHEMA_SQL, reset=True, assume_yes=True, log=QUIET)
        trunc = next(s for s in conn.sqls() if s.startswith("TRUNCATE"))
        self.assertIn("RESTART IDENTITY CASCADE", trunc)
        names = trunc.replace("TRUNCATE TABLE ", "").split(" RESTART")[0].split(", ")
        self.assertEqual(names[0], "churn_predictions")
        self.assertEqual(names[-1], "customers")
        self.assertFalse(any("CREATE TABLE" in s for s in conn.sqls()))   # existing tables are kept
        self.assertEqual(conn.kinds().count("commit"), 1)

    def test_existing_empty_tables_are_loaded_without_reset(self):
        conn = FakeConn(tables_present=L.TABLES, counts={})
        L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        self.assertEqual(conn.kinds().count("commit"), 1)

    def test_utc_timezone_is_set_before_any_copy(self):
        conn = FakeConn()
        L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        idx_tz = next(i for i, e in enumerate(conn.events) if e == ("sql", "SET LOCAL TIME ZONE 'UTC'"))
        idx_copy = next(i for i, e in enumerate(conn.events) if e[0] == "copy")
        self.assertLess(idx_tz, idx_copy)

    OLD7 = ["customers", "subscriptions", "usage_logs", "support_tickets", "transactions",
            "model_registry", "churn_predictions"]
    MIG = {"feature_snapshots": "CREATE TABLE IF NOT EXISTS feature_snapshots (customer_id VARCHAR(20));"}

    def test_old_seven_table_database_gets_migrated_then_loaded(self):
        conn = FakeConn(tables_present=self.OLD7, counts={})
        L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET, migrations=self.MIG)
        sqls = conn.sqls()
        self.assertIn(self.MIG["feature_snapshots"], sqls)
        self.assertFalse(any("CREATE TABLE customers" in s for s in sqls))      # existing tables untouched
        idx_mig = next(i for i, e in enumerate(conn.events) if e == ("sql", self.MIG["feature_snapshots"]))
        idx_copy = next(i for i, e in enumerate(conn.events) if e[0] == "copy")
        self.assertLess(idx_mig, idx_copy)                                      # table exists before it is filled
        self.assertEqual(conn.kinds().count("commit"), 1)

    def test_migration_of_populated_database_without_reset_rolls_back_the_migration(self):
        conn = FakeConn(tables_present=self.OLD7, counts={"customers": 7043})
        with self.assertRaises(SystemExit):
            L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET, migrations=self.MIG)
        self.assertIn(self.MIG["feature_snapshots"], conn.sqls())           # it ran inside the transaction...
        self.assertNotIn("commit", conn.kinds())                            # ...and was never committed
        self.assertEqual(conn.kinds()[-1], "rollback")

    def test_populated_old_database_with_reset_migrates_wipes_and_loads(self):
        conn = FakeConn(tables_present=self.OLD7, counts={"customers": 7043})
        L.run_load(conn, self.frames, SCHEMA_SQL, reset=True, assume_yes=True, log=QUIET, migrations=self.MIG)
        sqls = conn.sqls()
        self.assertLess(sqls.index(self.MIG["feature_snapshots"]), next(i for i, s in enumerate(sqls) if s.startswith("TRUNCATE")))
        self.assertIn("feature_snapshots", next(s for s in sqls if s.startswith("TRUNCATE")))
        self.assertEqual(conn.kinds().count("commit"), 1)

    def test_old_database_without_a_migration_is_refused(self):
        conn = FakeConn(tables_present=self.OLD7)
        with self.assertRaises(SystemExit) as cm:
            L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET, migrations={})
        self.assertIn("no migration", str(cm.exception))
        self.assertNotIn("commit", conn.kinds())

    def test_partial_schema_is_refused(self):
        conn = FakeConn(tables_present=["customers", "subscriptions"])
        with self.assertRaises(SystemExit):
            L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        self.assertNotIn("commit", conn.kinds())

    def test_failure_midway_rolls_back_and_never_commits(self):
        conn = FakeConn(fail_on=["COPY transactions"])
        with self.assertRaises(RuntimeError):
            L.run_load(conn, self.frames, SCHEMA_SQL, log=QUIET)
        self.assertNotIn("commit", conn.kinds())
        self.assertEqual(conn.kinds()[-1], "rollback")
        loaded = [e[1].split()[1] for e in conn.events if e[0] == "copy"]
        self.assertNotIn("churn_predictions", loaded)                  # stopped before the later tables

    def test_vector_extension_falls_back_to_default_schema(self):
        conn = FakeConn(script=[("FROM pg_extension", [])], fail_on=["WITH SCHEMA extensions"])
        msg = L.ensure_vector_extension(conn)
        self.assertIn("default schema", msg)
        self.assertFalse(conn.autocommit)                              # restored

    def test_vector_extension_failure_gives_actionable_error(self):
        conn = FakeConn(script=[("FROM pg_extension", [])], fail_on=["CREATE EXTENSION"])
        with self.assertRaises(SystemExit) as cm:
            L.ensure_vector_extension(conn)
        self.assertIn("Extensions", str(cm.exception))
        self.assertFalse(conn.autocommit)


class TestVerify(Base):
    def conn(self, counts, rls=True, bad_json=0, amount=70.5, charge=70.49, no_snap=0):
        script = [
            ("LEFT JOIN feature_snapshots", [(no_snap,)]),
            ("GROUP BY split", [("test", 1, "2026-08-29", "2026-08-29"), ("train", 1, "2026-09-01", "2026-09-01")]),
            ("jsonb_typeof", [(bad_json,)]),
            ("GROUP BY model_id", [("xgb_churn_v2", 2, 2, 2), ("lstm_churn_v1", 2, 0, 0)]),
            ("relrowsecurity", [(t, rls) for t in L.TABLES]),
            ("pg_database_size", [("9 MB",)]),
            ("ORDER BY p.churn_probability", [("C1", 0.91, "num__tenure_months")]),
            ("SUM(amount)", [(amount,)]),
            ("SUM(monthly_charge)", [(charge,)]),
        ]
        return FakeConn(counts=counts, script=script)

    def good_counts(self):
        return {t: len(self.frames[t]) for t in L.TABLES}

    def test_passes_when_everything_matches(self):
        self.assertTrue(L.verify(self.conn(self.good_counts()), L.expected_checks(self.frames), log=QUIET))

    def test_fails_on_count_mismatch(self):
        c = self.good_counts()
        c["usage_logs"] -= 1
        self.assertFalse(L.verify(self.conn(c), L.expected_checks(self.frames), log=QUIET))

    def test_fails_when_rls_off(self):
        self.assertFalse(L.verify(self.conn(self.good_counts(), rls=False), L.expected_checks(self.frames), log=QUIET))

    def test_fails_when_customers_lack_snapshots_but_only_on_full_load(self):
        self.assertFalse(L.verify(self.conn(self.good_counts(), no_snap=2), L.expected_checks(self.frames), log=QUIET))
        self.assertTrue(L.verify(self.conn(self.good_counts(), no_snap=2), None, log=QUIET))   # verify-only: info only

    def test_fails_on_sum_mismatch_and_bad_json(self):
        self.assertFalse(L.verify(self.conn(self.good_counts(), amount=1.0), L.expected_checks(self.frames), log=QUIET))
        self.assertFalse(L.verify(self.conn(self.good_counts(), bad_json=3), L.expected_checks(self.frames), log=QUIET))


class TestConnectionUrl(unittest.TestCase):
    def test_placeholder_and_missing_are_rejected(self):
        for bad in ("", None, "postgresql://postgres:[YOUR-PASSWORD]@db.x.supabase.co:5432/postgres"):
            with self.assertRaises(SystemExit):
                L.normalize_url(bad)

    def test_supabase_gets_sslmode_localhost_does_not(self):
        u = L.normalize_url("postgresql://postgres.abc:pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres")
        self.assertTrue(u.endswith("sslmode=require"))
        self.assertEqual(L.normalize_url(u), u)                          # idempotent
        loc = "postgresql://postgres:pw@localhost:5432/kairos"
        self.assertEqual(L.normalize_url(loc), loc)

    def test_password_is_masked(self):
        m = L.masked("postgresql://postgres.abc:SuperSecret1@host.supabase.com:5432/postgres")
        self.assertNotIn("SuperSecret1", m)
        self.assertIn("***", m)


if __name__ == "__main__":
    unittest.main()
