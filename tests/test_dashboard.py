"""
Tests for the interactive-dashboard analytics and endpoints.

    python -m unittest tests.test_dashboard -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api import dashboard as D  # noqa: E402
from tests.test_api import ApiTestBase  # noqa: E402
from tests.test_recommend import ctx, drv  # noqa: E402


class TestValidationPayload(unittest.TestCase):
    def test_perfect_ranking(self):
        v = D.validation_payload([1, 1, 0, 0], [0.9, 0.8, 0.3, 0.2], deciles=2)
        self.assertAlmostEqual(v["roc_auc"], 1.0)
        t = v["thresholds"][50]                      # threshold 0.50
        self.assertEqual((t["tp"], t["fp"], t["fn"], t["tn"]), (2, 0, 0, 2))
        self.assertEqual(t["precision"], 1.0)
        self.assertEqual(v["deciles"][0]["actual_churn_rate"], 1.0)
        self.assertAlmostEqual(v["deciles"][0]["lift"], 2.0)       # base rate 0.5
        self.assertAlmostEqual(v["deciles"][-1]["cum_churners_captured"], 1.0)

    def test_ties_give_auc_half(self):
        self.assertAlmostEqual(D.validation_payload([1, 0], [0.5, 0.5], deciles=2)["roc_auc"], 0.5)

    def test_threshold_counts_add_up(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        p = np.clip(y * 0.3 + rng.random(400) * 0.7, 0, 1)
        v = D.validation_payload(y, p)
        for t in v["thresholds"]:
            self.assertEqual(t["tp"] + t["fp"] + t["fn"] + t["tn"], 400)
        recalls = [t["recall"] for t in v["thresholds"]]
        self.assertTrue(all(a >= b for a, b in zip(recalls, recalls[1:])))   # recall never rises with threshold
        self.assertEqual(v["thresholds"][0]["recall"], 1.0)

    def test_needs_both_classes(self):
        with self.assertRaises(ValueError):
            D.validation_payload([1, 1], [0.9, 0.8])

    def test_tiebreak_orders_equal_scores(self):
        y, p = [0, 1, 0, 1], [0.9, 0.9, 0.1, 0.1]
        a = D.validation_payload(y, p, deciles=2, tiebreak=[0.2, 0.8, 0.1, 0.05])
        self.assertEqual(a["deciles"][0]["actual_churn_rate"], 0.5)


class TestFilters(unittest.TestCase):
    def setUp(self):
        self.pop = [ctx("A-1", region="West", contract_type="Month-to-month", tenure_months=3, risk_tier="high"),
                    ctx("B-2", region="East", contract_type="One year", tenure_months=30, risk_tier="low"),
                    ctx("A-3", region="West", contract_type="One year", tenure_months=70, risk_tier="medium")]

    def test_filters(self):
        f = D.filter_contexts
        self.assertEqual([c["customer_id"] for c in f(self.pop, q="a-")], ["A-1", "A-3"])
        self.assertEqual(len(f(self.pop, tiers=["high", "low"])), 2)
        self.assertEqual(len(f(self.pop, regions=["West"], contracts=["One year"])), 1)
        self.assertEqual([c["customer_id"] for c in f(self.pop, bands=["0-6", "49+"])], ["A-1", "A-3"])

    def test_tenure_band_edges(self):
        self.assertEqual([D.tenure_band(m) for m in (0, 6, 7, 12, 13, 24, 25, 48, 49, None)],
                         ["0-6", "0-6", "7-12", "7-12", "13-24", "13-24", "25-48", "25-48", "49+", "unknown"])

    def test_aggregates(self):
        for c in self.pop:
            c["drivers"] = [drv("tenure_months")]
        a = D.aggregates(self.pop, {})
        self.assertEqual(a["customers"], 3)
        self.assertEqual(a["tiers"], {"high": 1, "medium": 1, "low": 1})
        self.assertEqual(sum(a["probability_histogram"]), 3)
        self.assertEqual(a["revenue_exposed_monthly"], 100.0)       # only the high-risk customer's charge


class TestBudgetCurve(unittest.TestCase):
    def test_spend_never_decreases_with_budget(self):
        pop = [ctx(f"C{i}", p=0.9 - i * 0.05, drivers=[drv("failed_payment_count_90d")], failed_payments_90d=1) for i in range(10)]
        curve = D.budget_curve(pop, 60, 6, 12, 1.0)
        self.assertEqual(len(curve), 6)
        spends = [c["spent"] for c in curve]
        self.assertTrue(all(a <= b for a, b in zip(spends, spends[1:])))
        for c in curve:
            self.assertLessEqual(c["spent"], c["budget"] + 1e-9)


class TestDashboardApi(ApiTestBase):
    def test_customers(self):
        j = self.client.get("/api/v1/customers").get_json()
        self.assertEqual(j["total"], 1)
        self.assertEqual(j["items"][0]["customer_id"], "C1")
        self.assertEqual(j["items"][0]["region"], "West")
        self.assertEqual(j["aggregates"]["tiers"]["high"], 1)
        self.assertEqual(self.client.get("/api/v1/customers?risk_tier=low").get_json()["total"], 0)
        self.assertEqual(self.client.get("/api/v1/customers?q=zzz").get_json()["items"], [])
        self.assertEqual(self.client.get("/api/v1/customers?region=West&tenure_band=13-24").get_json()["total"], 1)

    def test_customers_validation(self):
        for q in ("risk_tier=bogus", "sort=bogus", "order=sideways", "page_size=500", "page=0"):
            self.assertError(self.client.get("/api/v1/customers?" + q), 400, "VALIDATION_ERROR")

    def test_validation_unavailable_with_one_row(self):
        self.assertError(self.client.get("/api/v1/model/validation"), 503, "VALIDATION_UNAVAILABLE")

    def test_curve(self):
        j = self.client.get("/api/v1/campaign/curve?max_budget=100&points=4").get_json()
        self.assertEqual(len(j["curve"]), 4)
        self.assertError(self.client.get("/api/v1/campaign/curve?points=1"), 400, "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
