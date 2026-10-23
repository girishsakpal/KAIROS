"""
Backtest of the retention plan on the held-out test customers (the only ones with a known outcome).

    python -m src.recommend.backtest

For the same budget it compares three ways of choosing who to contact:
  random       random customers, one-size-fits-all general check-in
  top_risk     highest churn probability first, one-size-fits-all general check-in
  kairos       the Retention Action Engine plan (best action per customer, value-for-money order)

WHAT THIS MEASURES (and what it does not)
  churners_reached / churn_revenue_reached use the REAL outcome (actual_churn): how many customers who
  really left, and how much 12-month revenue, were contacted. This is empirical.
  expected_net_value uses the ASSUMED save rates in actions.py. There is no intervention history in the
  data, so this backtest cannot prove an action would have kept anyone; it shows where each policy
  spends the budget, not what the causal effect would have been.
"""
import json
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.config import Config  # noqa: E402
from src.api.repository import CsvRepository  # noqa: E402
from src.recommend import actions as A  # noqa: E402
from src.recommend import engine  # noqa: E402

HORIZON = 12


def _score(chosen_ids, ctxs, actual, spent, exp_net, exp_saved):
    reached = [c for c in chosen_ids if actual.get(c) == 1]
    return {"customers_contacted": len(chosen_ids), "spent": round(spent, 2),
            "churners_reached": len(reached),
            "share_of_all_churners_reached": round(len(reached) / max(1, sum(actual.values())), 3),
            "churn_revenue_reached": round(sum(ctxs[c]["monthly_charge"] * HORIZON for c in reached), 2),
            "precision_of_contacts": round(len(reached) / max(1, len(chosen_ids)), 3),
            "assumed_expected_revenue_saved": round(exp_saved, 2), "assumed_expected_net_value": round(exp_net, 2)}


def _generic(order, ctxs, actual, budget):
    act = A.BY_ID["general_checkin"]
    spent, ids, saved = 0.0, [], 0.0
    for cid in order:
        cost = act.cost_for(ctxs[cid])
        if spent + cost > budget:
            break
        spent += cost
        ids.append(cid)
        saved += ctxs[cid]["churn_probability"] * act.base_save_rate * ctxs[cid]["monthly_charge"] * HORIZON
    return _score(ids, ctxs, actual, spent, saved - spent, saved)


def run(budgets=(500, 1000, 2000, 4000), seeds=200):
    cfg = dict(Config.__dict__)
    repo = CsvRepository(cfg)
    ids = repo.scored_customer_ids()
    ctxs = {c: engine.build_context(repo, c) for c in ids}
    pred = pd.read_csv(Path(cfg["MODELS_DIR"]) / cfg["PREDICTIONS_FILE"]).drop_duplicates("customer_id", keep="last")
    actual = dict(zip(pred.customer_id, pred.actual_churn.astype(int)))
    by_risk = sorted(ids, key=lambda c: ctxs[c]["churn_probability"], reverse=True)
    out = {"test_customers": len(ids), "test_churners": int(sum(actual.values())), "horizon_months": HORIZON,
           "note": "churners_reached uses real outcomes; expected values use ASSUMED save rates.", "results": []}
    for b in budgets:
        k = engine.plan_campaign(list(ctxs.values()), b, HORIZON)
        kres = _score([s["customer_id"] for s in k["selected"]], ctxs, actual,
                      k["summary"]["spent"], k["summary"]["expected_net_value"], k["summary"]["expected_revenue_saved"])
        rnd = []
        for s in range(seeds):
            order = ids[:]
            random.Random(s).shuffle(order)
            rnd.append(_generic(order, ctxs, actual, b))
        avg = {key: round(sum(r[key] for r in rnd) / seeds, 3) for key in rnd[0]}
        out["results"].append({"budget": b, "random": avg, "top_risk": _generic(by_risk, ctxs, actual, b), "kairos": kres})
    return out


if __name__ == "__main__":
    res = run()
    path = Path(Config.MODELS_DIR) / "retention_backtest.json"
    path.write_text(json.dumps(res, indent=2))
    print(f"{res['test_customers']} test customers, {res['test_churners']} real churners\n")
    for r in res["results"]:
        print(f"Budget {r['budget']}")
        for name in ("random", "top_risk", "kairos"):
            x = r[name]
            print(f"  {name:9s} contacted={x['customers_contacted']:6.0f} churners_reached={x['churners_reached']:6.0f} "
                  f"({x['share_of_all_churners_reached']:.0%}) churn_rev={x['churn_revenue_reached']:10.0f} "
                  f"precision={x['precision_of_contacts']:.2f} assumed_net={x['assumed_expected_net_value']:9.0f}")
    print(f"\nsaved {path}")
