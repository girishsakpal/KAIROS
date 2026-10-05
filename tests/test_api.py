"""
API tests (Phase 3 skeleton). Uses a tiny hand-built fixture dataset, NOT the real
repo data, so every expected number can be verified by hand.

Run from the repo root (either works):
    python -m unittest tests.test_api -v
    python -m pytest tests/test_api.py -v      # pytest collects unittest classes
"""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.api.app import create_app  # noqa: E402

NOW = datetime(2026, 10, 5, tzinfo=timezone.utc)
TOP = [
    {"feature": "num__failed_payment_count_90d", "shap_value": 0.41, "direction": "increases_risk"},
    {"feature": "cat__contract_type_Month-to-month", "shap_value": 0.3, "direction": "increases_risk"},
    {"feature": "num__tenure_months", "shap_value": -0.18, "direction": "decreases_risk"},
]


def _dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def build_fixture(root: Path, with_model=True):
    syn, proc, mod = root / "syn", root / "proc", root / "models"
    for d in (syn, proc, mod):
        d.mkdir()
    # C1: active, scored.  C2: churned, NOT scored.  C3: not in features (fallback snapshot), not scored.
    pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"], "signup_date": ["2025-02-14", "2024-01-01", "2026-01-01"],
        "region": ["West", "East", "North"], "signup_channel": ["organic", "referral", "social"],
        "age": [40, 30, 22], "gender": ["Male", "Female", "Male"]}).to_csv(syn / "customers.csv", index=False)
    pd.DataFrame({
        "customer_id": ["C1", "C2", "C3"], "plan_type": ["Pro", "Basic", "Basic"],
        "contract_type": ["Two year", "Month-to-month", "Month-to-month"],
        "billing_cycle": ["Annual", "Monthly", "Monthly"], "payment_method": ["Credit card", "Mailed check", "Mailed check"],
        "monthly_charge": [49.99, 20.0, 30.0], "tenure_months": [19, 5, 3],
        "status": ["active", "cancelled", "active"]}).to_csv(syn / "subscriptions.csv", index=False)
    # C1 usage: Mar..Aug = 20,18,16,10,8,6 (valid) + Oct=99 (AFTER snapshot, must be ignored)
    months = ["2026-03-01", "2026-04-01", "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01", "2026-10-01"]
    pd.DataFrame({"customer_id": ["C1"] * 7, "log_month": months,
                  "session_count": [20, 18, 16, 10, 8, 6, 99]}).to_csv(syn / "usage_logs.csv", index=False)
    # C1 tickets: t1 billing open 08-20 | t2 technical closed 06-01 | t3 AFTER snapshot (ignored)
    pd.DataFrame({"customer_id": ["C1"] * 3, "created_at": ["2026-08-20", "2026-06-01", "2026-09-15"],
                  "status": ["open", "closed", "open"],
                  "category": ["billing", "technical", "other"]}).to_csv(syn / "support_tickets.csv", index=False)
    # C1 txns: successes 06-01, 07-01, 08-04 (=150 lifetime) | failed 08-01 (in 90d), failed 05-01 (outside)
    # | success 09-10 (AFTER snapshot, ignored)
    pd.DataFrame({"customer_id": ["C1"] * 6,
                  "transaction_date": ["2026-06-01", "2026-07-01", "2026-08-01", "2026-05-01", "2026-08-04", "2026-09-10"],
                  "amount": [50] * 6, "transaction_type": ["payment", "payment", "failed_payment", "failed_payment", "payment", "payment"],
                  "status": ["success", "success", "failed", "failed", "success", "success"]}).to_csv(syn / "transactions.csv", index=False)
    pd.DataFrame({"customer_id": ["C1"], "snapshot_date": ["2026-09-01"]}).to_csv(proc / "features_train.csv", index=False)
    pd.DataFrame({"customer_id": ["C2"], "snapshot_date": ["2026-08-29"]}).to_csv(proc / "features_test.csv", index=False)

    if with_model:
        pd.DataFrame({"customer_id": ["C1"], "model_id": ["xgb_churn_v2"], "predicted_at": ["2026-10-05T01:00:00+00:00"],
                      "churn_probability": [0.8123456], "risk_tier": ["high"], "actual_churn": [1],
                      "top_features": [json.dumps(TOP)]}).to_csv(mod / "xgboost_predictions_test.csv", index=False)
        base = {"algorithm": "XGBoost", "trained_at": "2026-10-03T00:00:00+00:00", "roc_auc": 0.88, "pr_auc": 0.84,
                "calibration_error": 0.14, "drift_psi_score": None, "drift_ks_pvalue": None, "drift_detected": False}
        _dump({**base, "model_id": "xgb_churn_v2", "status": "challenger"}, mod / "xgboost_registry_row.json")
        _dump({**base, "model_id": "lstm_churn_v1", "algorithm": "LSTM", "status": "champion", "roc_auc": 0.9158},
              mod / "model_registry_row.json")
    return {"SYNTHETIC_DIR": syn, "PROCESSED_DIR": proc, "MODELS_DIR": mod, "NOW": NOW}


class ApiTestBase(unittest.TestCase):
    with_model = True
    extra = {}

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        cfg = build_fixture(Path(self._tmp.name), self.with_model)
        self.client = create_app({**cfg, **self.extra, "TESTING": True}).test_client()

    def assertError(self, resp, status, code):
        self.assertEqual(resp.status_code, status, resp.get_data(as_text=True))
        err = resp.get_json()["error"]
        self.assertEqual(err["code"], code)
        self.assertIn("message", err)
        self.assertIn("details", err)


class TestPredict(ApiTestBase):
    def post(self, body):
        return self.client.post("/api/v1/predict", json=body)

    def test_happy_path(self):
        r = self.post({"customer_id": "C1"})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertEqual(j["customer_id"], "C1")
        self.assertEqual(j["churn_probability"], 0.8123)
        self.assertEqual(j["risk_tier"], "high")
        self.assertEqual(j["model_id"], "xgb_churn_v2")
        self.assertEqual(j["model_status"], "challenger")
        self.assertFalse(j["recomputed"])
        # encoder prefixes stripped, order and values preserved
        self.assertEqual([d["feature"] for d in j["top_features"]],
                         ["failed_payment_count_90d", "contract_type_Month-to-month", "tenure_months"])
        self.assertEqual(j["top_features"][2]["direction"], "decreases_risk")

    def test_drift_status_not_stale_and_unchecked(self):
        ds = self.post({"customer_id": "C1"}).get_json()["drift_status"]
        self.assertEqual(ds["days_since_training"], 2)       # 2026-10-03 -> 2026-10-05
        self.assertFalse(ds["is_stale"])
        self.assertFalse(ds["drift_checked"])                # PSI/KS not built yet
        self.assertEqual(ds["drifted_features"], [])

    def test_unknown_customer(self):
        self.assertError(self.post({"customer_id": "NOPE"}), 404, "NOT_FOUND")

    def test_known_customer_without_prediction(self):
        self.assertError(self.post({"customer_id": "C2"}), 404, "PREDICTION_NOT_FOUND")

    def test_validation(self):
        self.assertError(self.post({}), 400, "VALIDATION_ERROR")
        self.assertError(self.post({"customer_id": ""}), 400, "VALIDATION_ERROR")
        self.assertError(self.post({"customer_id": 123}), 400, "VALIDATION_ERROR")
        self.assertError(self.post({"customer_id": "C1", "force_recompute": "yes"}), 400, "VALIDATION_ERROR")
        self.assertError(self.client.post("/api/v1/predict", data="not json"), 400, "VALIDATION_ERROR")
        self.assertError(self.post([1, 2]), 400, "VALIDATION_ERROR")

    def test_force_recompute_not_implemented(self):
        self.assertError(self.post({"customer_id": "C1", "force_recompute": True}), 501, "NOT_IMPLEMENTED")


class TestPredictStale(ApiTestBase):
    extra = {"STALE_AFTER_DAYS": 1}

    def test_stale_flag_follows_threshold(self):
        ds = self.client.post("/api/v1/predict", json={"customer_id": "C1"}).get_json()["drift_status"]
        self.assertTrue(ds["is_stale"])


class TestPredictNoModel(ApiTestBase):
    with_model = False

    def test_503_when_no_model(self):
        r = self.client.post("/api/v1/predict", json={"customer_id": "C1"})
        self.assertError(r, 503, "MODEL_UNAVAILABLE")

    def test_unknown_customer_still_404_first(self):
        r = self.client.post("/api/v1/predict", json={"customer_id": "NOPE"})
        self.assertError(r, 404, "NOT_FOUND")


class TestCustomerDetail(ApiTestBase):
    def get(self, cid, q=""):
        return self.client.get(f"/api/v1/customer-detail/{cid}{q}")

    def test_full_view_with_hand_checked_numbers(self):
        j = self.get("C1").get_json()
        self.assertEqual(j["profile"]["plan"], "Pro Annual")
        self.assertEqual(j["profile"]["region"], "West")
        self.assertEqual(j["subscription"]["status"], "active")
        # usage: valid months 20,18,16,10,8,6 ; Oct=99 is after snapshot 2026-09-01 -> ignored
        self.assertEqual(j["usage_summary"], {"sessions_30d": 6, "sessions_90d_avg": 8.0, "trend": "declining"})
        # support: t3 (09-15) is after snapshot -> ignored; t1 open; tickets_60d = t1 only
        self.assertEqual(j["support_summary"], {"open_tickets": 1, "tickets_60d": 1, "last_ticket_category": "billing"})
        # txns: 09-10 ignored; failed in 90d = 08-01 only; LTV = 3 successes x 50
        self.assertEqual(j["transaction_summary"], {"failed_payments_90d": 1, "lifetime_value": 150.0})
        lp = j["latest_prediction"]
        self.assertEqual(lp["top_features"][0], "failed_payment_count_90d")
        self.assertEqual(lp["drift_status"], {"is_stale": False, "days_since_training": 2})

    def test_customer_with_no_history(self):
        j = self.get("C3").get_json()          # no snapshot -> reference-date fallback, no rows
        self.assertEqual(j["usage_summary"]["trend"], "unknown")
        self.assertEqual(j["support_summary"], {"open_tickets": 0, "tickets_60d": 0, "last_ticket_category": None})
        self.assertEqual(j["transaction_summary"], {"failed_payments_90d": 0, "lifetime_value": 0.0})
        self.assertIsNone(j["latest_prediction"])

    def test_include_filters_sections(self):
        j = self.get("C1", "?include=support,prediction").get_json()
        self.assertEqual(set(j), {"customer_id", "profile", "support_summary", "latest_prediction"})

    def test_bad_include(self):
        self.assertError(self.get("C1", "?include=support,bogus"), 400, "VALIDATION_ERROR")
        self.assertError(self.get("C1", "?include="), 400, "VALIDATION_ERROR")

    def test_unknown_customer(self):
        self.assertError(self.get("NOPE"), 404, "NOT_FOUND")


class TestModelMetricsAndGeneral(ApiTestBase):
    def test_model_metrics_shape(self):
        j = self.client.get("/api/v1/model-metrics").get_json()
        self.assertEqual(j["champion_model"]["model_id"], "lstm_churn_v1")
        self.assertEqual(j["serving_model"]["model_id"], "xgb_churn_v2")
        self.assertEqual(j["champion_model"]["metrics"]["roc_auc"], 0.9158)
        self.assertIsNone(j["evaluation_harness"]["ragas_faithfulness"])
        self.assertEqual(j["history"], [])

    def test_model_metrics_requested_and_history(self):
        j = self.client.get("/api/v1/model-metrics?model_id=xgb_churn_v2&include_history=true").get_json()
        self.assertEqual(j["requested_model"]["model_id"], "xgb_churn_v2")
        self.assertEqual(len(j["history"]), 2)
        self.assertError(self.client.get("/api/v1/model-metrics?model_id=zzz"), 404, "NOT_FOUND")

    def test_health(self):
        j = self.client.get("/api/v1/health").get_json()
        self.assertEqual(j, {"status": "ok", "serving_model_id": "xgb_churn_v2", "customers": 3, "predictions": 1})

    def test_unknown_route_and_wrong_method_use_error_format(self):
        self.assertError(self.client.get("/api/v1/segment"), 404, "NOT_FOUND")
        self.assertError(self.client.get("/api/v1/predict"), 405, "METHOD_NOT_ALLOWED")

    def test_auth_placeholder_marks_roles(self):
        # Roles from the contract are attached now, so Phase 7 only has to enforce them.
        fn = self.client.application.view_functions["v1.predict"]
        self.assertEqual(fn.required_roles, ("ops", "admin"))


if __name__ == "__main__":
    unittest.main()
