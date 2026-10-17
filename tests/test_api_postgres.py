"""
Tests for the Postgres-backed API repository, using SQLite as a stand-in database.

What this proves: the repository's SQL and Python logic produce the SAME responses as the CSV-backed
repository for the same data (shared summaries, model selection, 404s, time-window rules).
What it does NOT prove: Postgres-specific behaviour (types, jsonb). That is what
`python -m src.api.check_parity` is for, against your real Supabase database.

    python -m unittest tests.test_api_postgres -v
"""
import json
import os
import sqlite3
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.check_parity import compare_apps, diff  # noqa: E402
from src.api.repository_pg import PostgresRepository  # noqa: E402
from tests.test_api import NOW, build_fixture  # noqa: E402


class OperationalError(Exception):
    """Same class NAME as psycopg2.OperationalError (the repository matches on the name)."""


class SqliteCursor:
    def __init__(self, conn):
        self.cur = conn.cursor()

    def execute(self, sql, params=()):
        params = [p.isoformat() if hasattr(p, "isoformat") else p for p in params]
        self.cur.execute(sql.replace("%s", "?"), params)

    @property
    def description(self):
        return self.cur.description

    def fetchall(self):
        return self.cur.fetchall()


class SqliteConn:
    def __init__(self, raw):
        self.raw = raw

    def cursor(self):
        return SqliteCursor(self.raw)

    def rollback(self):
        self.raw.rollback()


def build_sqlite(cfg, extra_lstm_prediction=True):
    """Load the test_api fixture files into an in-memory SQLite database shaped like the Postgres tables."""
    raw = sqlite3.connect(":memory:", check_same_thread=False)
    raw.create_function("jsonb_array_length", 1, lambda s: len(json.loads(s)))
    syn, proc, mod = Path(cfg["SYNTHETIC_DIR"]), Path(cfg["PROCESSED_DIR"]), Path(cfg["MODELS_DIR"])
    for t in ("customers", "subscriptions", "usage_logs", "support_tickets", "transactions"):
        pd.read_csv(syn / f"{t}.csv").to_sql(t, raw, index=False)
    snaps = pd.concat([pd.read_csv(proc / "features_train.csv").assign(split="train"),
                       pd.read_csv(proc / "features_test.csv").assign(split="test")])
    snaps[["customer_id", "snapshot_date", "split"]].to_sql("feature_snapshots", raw, index=False)
    reg = pd.DataFrame([json.loads((mod / f).read_text())
                        for f in ("xgboost_registry_row.json", "model_registry_row.json")])
    reg["model_name"] = reg["model_id"]
    reg["drift_detected"] = reg["drift_detected"].astype(int)
    reg[["model_id", "model_name", "algorithm", "trained_at", "roc_auc", "pr_auc", "calibration_error",
         "drift_psi_score", "drift_ks_pvalue", "drift_detected", "status"]].to_sql("model_registry", raw, index=False)
    preds = pd.read_csv(mod / "xgboost_predictions_test.csv")
    if extra_lstm_prediction:     # a NEWER prediction from the champion with no drivers must be ignored
        preds = pd.concat([preds, pd.DataFrame({
            "customer_id": ["C1"], "model_id": ["lstm_churn_v1"], "predicted_at": ["2026-10-06T00:00:00+00:00"],
            "churn_probability": [0.5], "risk_tier": ["medium"], "actual_churn": [1], "top_features": ["[]"]})])
    preds.to_sql("churn_predictions", raw, index=False)
    return raw


def pg_repo(cfg, raw=None, **kw):
    conn = SqliteConn(raw if raw is not None else build_sqlite(cfg))
    return PostgresRepository(lambda: conn, lambda c, broken=False: None, cfg["REFERENCE_DATE"],
                              model_cache_seconds=0, **kw)


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        fx = build_fixture(Path(self._tmp.name))
        self.cfg = {**create_app({**fx, "REPOSITORY": "csv"}).config}      # full resolved config
        self.csv_app = create_app({**fx, "REPOSITORY": "csv"})
        self.pg_app = create_app({**fx, "REPOSITORY": "csv"}, repository=pg_repo(self.cfg))


class TestParityWithCsvRepository(Base):
    def test_every_endpoint_identical_for_scored_unscored_and_unknown_customers(self):
        problems = compare_apps(self.csv_app.test_client(), self.pg_app.test_client(),
                                ["C1", "C2", "C3", "NOPE"], log=lambda *a: None)
        self.assertEqual(problems, [], problems[:5])

    def test_the_comparison_can_actually_fail(self):
        """Guard against a vacuous parity test: break the Postgres side and expect differences."""
        raw = build_sqlite(self.cfg)
        raw.execute("UPDATE usage_logs SET session_count = 1")
        raw.commit()      # the repository rolls back after every read, so uncommitted edits would vanish
        app = create_app({**self.cfg, "REPOSITORY": "csv"}, repository=pg_repo(self.cfg, raw))
        problems = compare_apps(self.csv_app.test_client(), app.test_client(), ["C1"], log=lambda *a: None)
        self.assertTrue(any("usage_summary" in p[2] for p in problems))


class TestPostgresRepositoryBehaviour(Base):
    def test_serving_model_is_derived_from_data_not_the_champion(self):
        repo = pg_repo(self.cfg)
        self.assertEqual(repo.serving_model()["model_id"], "xgb_churn_v2")      # has drivers
        self.assertEqual(repo.champion_model()["model_id"], "lstm_churn_v1")    # champion, no drivers

    def test_newer_prediction_from_another_model_is_ignored(self):
        p = pg_repo(self.cfg).latest_prediction("C1")
        self.assertEqual(p["model_id"], "xgb_churn_v2")
        self.assertEqual(len(p["top_features"]), 3)

    def test_no_model_with_drivers_gives_503_path(self):
        raw = build_sqlite(self.cfg)
        raw.execute("UPDATE churn_predictions SET top_features = '[]'")
        raw.commit()      # the repository rolls back after every read, so uncommitted edits would vanish
        repo = pg_repo(self.cfg, raw)
        self.assertIsNone(repo.serving_model())
        self.assertIsNone(repo.latest_prediction("C1"))
        app = create_app({**self.cfg, "REPOSITORY": "csv"}, repository=repo)
        r = app.test_client().post("/api/v1/predict", json={"customer_id": "C1"})
        self.assertEqual((r.status_code, r.get_json()["error"]["code"]), (503, "MODEL_UNAVAILABLE"))

    def test_snapshot_rule_still_applies_through_the_database(self):
        usage = pg_repo(self.cfg).usage_summary("C1")      # the Oct row (99 sessions) is after the snapshot
        self.assertEqual(usage, {"sessions_30d": 6, "sessions_90d_avg": 8.0, "trend": "declining"})

    def test_customer_without_snapshot_falls_back_to_reference_date(self):
        self.assertEqual(pg_repo(self.cfg).snapshot_date("C3"), pd.Timestamp(self.cfg["REFERENCE_DATE"]))

    def test_snapshot_lookup_is_cached(self):
        repo = pg_repo(self.cfg)
        calls = []
        orig = repo._rows
        repo._rows = lambda sql, params=(): (calls.append(sql), orig(sql, params))[1]
        repo.snapshot_date("C1"); repo.snapshot_date("C1")
        self.assertEqual(sum("feature_snapshots" in c for c in calls), 1)

    def test_counts(self):
        self.assertEqual(pg_repo(self.cfg).counts(), {"customers": 3, "predictions": 1})


class TestConnectionHandling(Base):
    def make(self, failures, error=OperationalError("server closed the connection")):
        raw = build_sqlite(self.cfg)
        good = SqliteConn(raw)
        log = {"released": [], "acquired": 0}

        class FlakyCursor(SqliteCursor):
            def execute(inner, sql, params=()):
                if log["acquired"] <= failures:
                    raise error
                super().execute(sql, params)

        class FlakyConn(SqliteConn):
            def cursor(inner):
                return FlakyCursor(raw)

        def acquire():
            log["acquired"] += 1
            return FlakyConn(raw) if log["acquired"] <= failures else good

        repo = PostgresRepository(acquire, lambda c, broken=False: log["released"].append(broken),
                                  self.cfg["REFERENCE_DATE"], model_cache_seconds=0)
        return repo, log

    def test_dropped_connection_is_retried_once_and_discarded(self):
        repo, log = self.make(failures=1)
        self.assertTrue(repo.customer_exists("C1"))
        self.assertEqual(log["released"], [True, False])      # bad one closed, good one returned

    def test_second_failure_is_raised(self):
        repo, log = self.make(failures=2)
        with self.assertRaises(OperationalError):
            repo.customer_exists("C1")
        self.assertEqual(log["released"], [True, True])

    def test_non_connection_errors_are_not_retried(self):
        repo, log = self.make(failures=1, error=ValueError("bad sql"))
        with self.assertRaises(ValueError):
            repo.customer_exists("C1")
        self.assertEqual(log["acquired"], 1)
        self.assertEqual(log["released"], [False])


class TestWiring(Base):
    """from_env and create_app wiring, with a stand-in connection pool (psycopg2 is not needed)."""

    URL = "postgresql://postgres.abc:pw@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"

    def test_from_env_builds_a_pool_from_database_url_and_releases_with_close_flag(self):
        seen = {}

        class FakePool:
            def __init__(self, minconn, maxconn, dsn, connect_timeout):
                seen.update(dsn=dsn, size=(minconn, maxconn))

            def getconn(self):
                return "CONN"

            def putconn(self, conn, close=False):
                seen["put"] = (conn, close)

        pool_mod = types.ModuleType("psycopg2.pool")
        pool_mod.ThreadedConnectionPool = FakePool
        pg_mod = types.ModuleType("psycopg2")
        pg_mod.pool = pool_mod
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *a, **k: None
        mods = {"psycopg2": pg_mod, "psycopg2.pool": pool_mod, "dotenv": dotenv_mod}
        with mock.patch.dict(sys.modules, mods), mock.patch.dict(os.environ, {"DATABASE_URL": self.URL}):
            repo = PostgresRepository.from_env("2026-09-16")
        self.assertTrue(seen["dsn"].endswith("sslmode=require"))           # Supabase host -> SSL enforced
        self.assertEqual(repo._acquire(), "CONN")
        repo._release("CONN", True)
        self.assertEqual(seen["put"], ("CONN", True))
        self.assertNotIn("pw", repo.description)                           # password never exposed

    def test_missing_database_url_is_a_clear_error(self):
        pool_mod = types.ModuleType("psycopg2.pool")
        pool_mod.ThreadedConnectionPool = object
        pg_mod = types.ModuleType("psycopg2")
        pg_mod.pool = pool_mod
        dotenv_mod = types.ModuleType("dotenv")
        dotenv_mod.load_dotenv = lambda *a, **k: None
        env = {k: v for k, v in os.environ.items() if k != "DATABASE_URL"}
        with mock.patch.dict(sys.modules, {"psycopg2": pg_mod, "psycopg2.pool": pool_mod, "dotenv": dotenv_mod}), \
                mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(SystemExit):
                PostgresRepository.from_env("2026-09-16")

    def test_create_app_chooses_backend_from_config(self):
        with mock.patch.object(PostgresRepository, "from_env", return_value="PG-REPO") as m:
            app = create_app({**self.cfg, "REPOSITORY": "postgres"})
        m.assert_called_once()
        self.assertEqual(app.extensions["kairos_repo"], "PG-REPO")
        with self.assertRaises(ValueError):
            create_app({**self.cfg, "REPOSITORY": "mongo"})


class TestDiffHelper(unittest.TestCase):
    def test_float_tolerance_and_real_differences(self):
        self.assertEqual(diff({"p": 0.81230}, {"p": 0.81235}), [])
        self.assertEqual(len(diff({"p": 0.8123}, {"p": 0.8200})), 1)

    def test_timestamps_compare_as_instants(self):
        self.assertEqual(diff({"t": "2026-10-03T00:00:00+00:00"}, {"t": "2026-10-03T05:30:00+05:30"}), [])
        self.assertEqual(len(diff({"t": "2026-10-03T00:00:00+00:00"}, {"t": "2026-10-04T00:00:00+00:00"})), 1)

    def test_structure_differences_and_bool_vs_number(self):
        self.assertEqual(len(diff({"a": 1}, {"a": 1, "b": 2})), 1)
        self.assertEqual(len(diff([1, 2], [1])), 1)
        self.assertEqual(len(diff({"x": True}, {"x": 1})), 1)       # bool is not silently equal to 1
        self.assertEqual(len(diff({"d": "2026-09-01"}, {"d": "2026-09-02"})), 1)


if __name__ == "__main__":
    unittest.main()
