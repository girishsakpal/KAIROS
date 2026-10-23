"""
Retention Action Engine tests. Pure-engine tests use hand-built contexts so every expectation can be
checked by hand; API tests reuse the tiny fixture from test_api (customer C1 is scored).

    python -m unittest tests.test_recommend -v
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.recommend import engine  # noqa: E402
from tests.test_api import ApiTestBase  # noqa: E402


def ctx(cid="X", p=0.9, charge=100.0, **kw):
    base = {"customer_id": cid, "churn_probability": p, "risk_tier": "high", "monthly_charge": charge,
            "tenure_months": 30, "contract_type": "One year", "payment_method": "Credit card (automatic)",
            "usage_trend": "stable", "open_tickets": 0, "failed_payments_90d": 0,
            "drivers": [{"feature": "age", "shap_value": 0.1, "direction": "increases_risk"}]}
    base.update(kw)
    return base


def drv(feature, shap=0.5, direction="increases_risk"):
    return {"feature": feature, "shap_value": shap, "direction": direction}


class TestRankActions(unittest.TestCase):
    def test_top_driver_beats_fact_only_match(self):
        c = ctx(drivers=[drv("failed_payment_count_90d")], failed_payments_90d=2, open_tickets=1)
        ranked = engine.rank_actions(c)
        pay = next(r for r in ranked if r["action_id"] == "payment_recovery")
        sup = next(r for r in ranked if r["action_id"] == "support_escalation")
        self.assertEqual(pay["match"], 1.0)             # model driver #1
        self.assertEqual(sup["match"], 0.6)             # customer fact only
        self.assertEqual(ranked[0]["action_id"], "payment_recovery")

    def test_guard_blocks_nonsense_actions(self):
        # model blames the month-to-month contract, but this customer is already on a one-year contract
        c = ctx(drivers=[drv("contract_type_Month-to-month")], contract_type="One year")
        self.assertNotIn("annual_contract_offer", [r["action_id"] for r in engine.rank_actions(c)])
        # long-tenure customer: onboarding call makes no sense even if tenure is a driver
        c = ctx(drivers=[drv("tenure_months")], tenure_months=40)
        self.assertNotIn("onboarding_call", [r["action_id"] for r in engine.rank_actions(c)])

    def test_risk_reducing_drivers_are_ignored(self):
        c = ctx(drivers=[drv("failed_payment_count_90d", -0.4, "decreases_risk")])
        self.assertNotIn("payment_recovery", [r["action_id"] for r in engine.rank_actions(c)])

    def test_money_arithmetic(self):
        c = ctx(p=0.8, charge=50.0, drivers=[drv("failed_payment_count_90d")], failed_payments_90d=1)
        r = next(x for x in engine.rank_actions(c, horizon_months=12) if x["action_id"] == "payment_recovery")
        # success = 0.35 * 1.0 ; at stake = 50*12 = 600 ; expected = 0.8*0.35*600 = 168 ; net = 168 - 4
        self.assertAlmostEqual(r["success_probability"], 0.35)
        self.assertAlmostEqual(r["expected_revenue_saved"], 168.0)
        self.assertAlmostEqual(r["net_value"], 164.0)

    def test_save_rate_scale_changes_value(self):
        c = ctx(drivers=[drv("failed_payment_count_90d")], failed_payments_90d=1)
        a = engine.rank_actions(c, save_rate_scale=1.0)[0]["expected_revenue_saved"]
        b = engine.rank_actions(c, save_rate_scale=0.5)[0]["expected_revenue_saved"]
        self.assertAlmostEqual(b, a / 2, places=1)

    def test_general_checkin_only_as_fallback(self):
        ids = [r["action_id"] for r in engine.rank_actions(ctx(charge=50.0))]
        self.assertEqual(ids, ["general_checkin"])
        c = ctx(drivers=[drv("failed_payment_count_90d")], failed_payments_90d=1)
        self.assertNotIn("general_checkin", [r["action_id"] for r in engine.rank_actions(c)])


class TestCampaign(unittest.TestCase):
    def setUp(self):
        self.pop = [ctx(f"C{i}", p=0.95 - i * 0.05, charge=60.0 + i,
                        drivers=[drv("failed_payment_count_90d")], failed_payments_90d=1) for i in range(8)]

    def test_budget_never_exceeded_and_one_action_per_customer(self):
        for budget in (4, 10, 17, 1000):
            plan = engine.plan_campaign(self.pop, budget)
            self.assertLessEqual(plan["summary"]["spent"], budget + 1e-9)
            ids = [s["customer_id"] for s in plan["selected"]]
            self.assertEqual(len(ids), len(set(ids)))

    def test_budget_below_cheapest_action_selects_nobody(self):
        self.assertEqual(engine.plan_campaign(self.pop, 1.0)["selected"], [])

    def test_max_contacts(self):
        self.assertEqual(len(engine.plan_campaign(self.pop, 1000, max_contacts=3)["selected"]), 3)

    def test_team_capacity_forces_fallback_or_skip(self):
        plan = engine.plan_campaign(self.pop, 1000, team_capacity={"Collections": 2})
        self.assertLessEqual(plan["summary"]["by_team"].get("Collections", 0), 2)

    def test_bigger_budget_never_reduces_expected_value(self):
        small = engine.plan_campaign(self.pop, 8)["summary"]["expected_net_value"]
        big = engine.plan_campaign(self.pop, 1000)["summary"]["expected_net_value"]
        self.assertGreaterEqual(big, small)


class TestRecommendApi(ApiTestBase):
    def test_recommend_c1(self):
        r = self.client.get("/api/v1/recommend/C1")
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        ids = [x["action_id"] for x in j["recommendations"]]
        self.assertEqual(ids[0], "payment_recovery")             # failed payments is driver #1
        self.assertIn("disclaimer", j)
        self.assertEqual(j["assumptions"]["revenue_at_stake"], round(49.99 * 12, 2))
        full = self.client.get("/api/v1/recommend/C1?limit=8").get_json()["recommendations"]
        full_ids = [x["action_id"] for x in full]
        self.assertNotIn("annual_contract_offer", full_ids)      # C1 is on a two-year contract
        self.assertNotIn("onboarding_call", full_ids)            # tenure 19 months

    def test_errors(self):
        self.assertError(self.client.get("/api/v1/recommend/NOPE"), 404, "NOT_FOUND")
        self.assertError(self.client.get("/api/v1/recommend/C2"), 404, "PREDICTION_NOT_FOUND")
        self.assertError(self.client.get("/api/v1/recommend/C1?limit=0"), 400, "VALIDATION_ERROR")
        self.assertError(self.client.get("/api/v1/recommend/C1?save_rate_scale=x"), 400, "VALIDATION_ERROR")

    def test_catalog(self):
        j = self.client.get("/api/v1/recommend/catalog").get_json()
        self.assertEqual(len(j["actions"]), 8)
        self.assertTrue(all("assumed_save_rate" in a for a in j["actions"]))

    def test_campaign(self):
        j = self.client.get("/api/v1/campaign?budget=100").get_json()
        self.assertEqual(j["summary"]["customers_contacted"], 1)
        self.assertEqual(j["selected"][0]["customer_id"], "C1")
        self.assertLessEqual(j["summary"]["spent"], 100)
        self.assertEqual(self.client.get("/api/v1/campaign?budget=1").get_json()["selected"], [])
        self.assertError(self.client.get("/api/v1/campaign?budget=-5"), 400, "VALIDATION_ERROR")


if __name__ == "__main__":
    unittest.main()
