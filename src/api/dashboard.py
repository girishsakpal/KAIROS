"""
Pure analytics behind the interactive dashboard (no Flask here, so it is easy to unit-test).

  filter_contexts     filter the scored customers by search text, tier, region, contract, tenure band
  aggregates          KPI numbers and chart data for a filtered set of customers
  validation_payload  ROC curve, threshold table, deciles and calibration from the held-out test set
  budget_curve        expected value of the campaign plan across a range of budgets
"""
import numpy as np

from ..recommend import actions as A
from ..recommend import engine

TENURE_BANDS = {"0-6": (0, 6), "7-12": (7, 12), "13-24": (13, 24), "25-48": (25, 48), "49+": (49, 10_000)}
SORTS = {"probability": lambda c: c["churn_probability"], "charge": lambda c: c["monthly_charge"],
         "tenure": lambda c: c["tenure_months"] or 0, "id": lambda c: c["customer_id"]}


def tenure_band(months):
    for name, (lo, hi) in TENURE_BANDS.items():
        if months is not None and lo <= months <= hi:
            return name
    return "unknown"


def filter_contexts(contexts, q=None, tiers=None, regions=None, contracts=None, bands=None):
    out = []
    ql = (q or "").strip().lower()
    for c in contexts:
        if ql and ql not in c["customer_id"].lower():
            continue
        if tiers and c["risk_tier"] not in tiers:
            continue
        if regions and c.get("region") not in regions:
            continue
        if contracts and c.get("contract_type") not in contracts:
            continue
        if bands and tenure_band(c.get("tenure_months")) not in bands:
            continue
        out.append(c)
    return out


def best_action(ctx, horizon=12, scale=1.0):
    ranked = engine.rank_actions(ctx, horizon, scale)
    return ranked[0] if ranked else None


def _counts(values):
    d = {}
    for v in values:
        d[v] = d.get(v, 0) + 1
    return d


def aggregates(ctxs, best):
    """best: dict customer_id -> best-action dict (or None)."""
    n = len(ctxs)
    tiers = _counts(c["risk_tier"] for c in ctxs)
    high = [c for c in ctxs if c["risk_tier"] == "high"]
    hist = [0] * 10
    for c in ctxs:
        hist[min(9, int(c["churn_probability"] * 10))] += 1
    regions = {}
    for c in ctxs:
        r = regions.setdefault(c.get("region") or "unknown", {"region": c.get("region") or "unknown", "customers": 0, "high": 0})
        r["customers"] += 1
        r["high"] += c["risk_tier"] == "high"
    drivers = _counts(A.label(next((d["feature"] for d in c["drivers"] if d["direction"] == "increases_risk"), "none"))
                      for c in ctxs if c["risk_tier"] == "high")
    acts = _counts(best[c["customer_id"]]["action"] for c in ctxs if best.get(c["customer_id"]))
    return {"customers": n, "tiers": {"high": tiers.get("high", 0), "medium": tiers.get("medium", 0), "low": tiers.get("low", 0)},
            "avg_probability": round(sum(c["churn_probability"] for c in ctxs) / n, 4) if n else 0,
            "revenue_exposed_monthly": round(sum(c["monthly_charge"] for c in high), 2),
            "probability_histogram": hist,
            "by_region": sorted(regions.values(), key=lambda r: -r["high"]),
            "top_driver_of_high_risk": [{"driver": k, "customers": v} for k, v in sorted(drivers.items(), key=lambda kv: -kv[1])[:8]],
            "recommended_action_mix": [{"action": k, "customers": v} for k, v in sorted(acts.items(), key=lambda kv: -kv[1])]}


def table_row(c, best_a):
    top = next((d for d in c["drivers"] if d["direction"] == "increases_risk"), None)
    return {"customer_id": c["customer_id"], "churn_probability": round(c["churn_probability"], 4),
            "risk_tier": c["risk_tier"], "region": c.get("region"), "plan": c.get("plan"),
            "contract_type": c.get("contract_type"), "monthly_charge": c["monthly_charge"],
            "tenure_months": c.get("tenure_months"), "top_driver": A.label(top["feature"]) if top else None,
            "best_action": best_a["action"] if best_a else None,
            "best_action_net_value": best_a["net_value"] if best_a else None}


def validation_payload(y, p, deciles=10, shap=None, tiebreak=None):
    """y: 0/1 outcomes, p: predicted probabilities (numpy arrays). tiebreak: optional second score used to
    order customers whose calibrated probability is identical (the raw model score), so deciles match the
    Power BI decile table."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        raise ValueError("need both churned and retained customers")
    order = np.lexsort((-np.asarray(tiebreak, dtype=float), -p)) if tiebreak is not None else np.argsort(-p, kind="stable")
    ys, ps = y[order], p[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    last = np.r_[ps[1:] != ps[:-1], True]               # one ROC point per distinct score
    tpr, fpr = np.r_[0, tps[last] / pos], np.r_[0, fps[last] / neg]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))   # trapezoid rule
    idx = np.unique(np.linspace(0, len(tpr) - 1, min(len(tpr), 120)).astype(int))
    thresholds = []
    for t in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        fn = pos - tp; tn = neg - fp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / pos
        thresholds.append({"threshold": float(t), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": round(prec, 4), "recall": round(rec, 4),
                           "f1": round(2 * prec * rec / (prec + rec), 4) if prec + rec else 0.0,
                           "accuracy": round((tp + tn) / len(y), 4)})
    base = pos / len(y)
    chunks = np.array_split(np.arange(len(ys)), deciles)
    dec, cum = [], 0
    for i, ch in enumerate(chunks, 1):
        ch_y, ch_p = ys[ch], ps[ch]
        cum += int(ch_y.sum())
        rate = float(ch_y.mean())
        dec.append({"decile": i, "customers": len(ch), "avg_predicted": round(float(ch_p.mean()), 4),
                    "actual_churn_rate": round(rate, 4), "lift": round(rate / base, 3),
                    "cum_churners_captured": round(cum / pos, 4)})
    bins = np.minimum((p * 10).astype(int), 9)
    calib = [{"bin": f"{b * 10}-{b * 10 + 10}%", "customers": int((bins == b).sum()),
              "avg_predicted": round(float(p[bins == b].mean()), 4), "actual_rate": round(float(y[bins == b].mean()), 4)}
             for b in range(10) if (bins == b).any()]
    return {"customers": len(y), "churners": pos, "base_rate": round(base, 4), "roc_auc": round(auc, 4),
            "roc": [{"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)} for i in idx],
            "thresholds": thresholds, "deciles": dec, "calibration": calib, "shap_global": shap or []}


def budget_curve(contexts, max_budget, points, horizon, scale):
    out = []
    for i in range(1, points + 1):
        b = max_budget * i / points
        s = engine.plan_campaign(contexts, b, horizon, scale)["summary"]
        out.append({"budget": round(b, 2), "spent": s["spent"], "customers_contacted": s["customers_contacted"],
                    "expected_revenue_saved": s["expected_revenue_saved"], "expected_net_value": s["expected_net_value"]})
    return out
