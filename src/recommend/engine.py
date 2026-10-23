"""
Retention Action Engine.

Turns a churn prediction + its SHAP drivers + a few customer facts into ranked, explainable retention
actions, and picks the best set of customers to contact under a budget.

IMPORTANT: this is a decision-support tool, not a guarantee. The save rates and costs in actions.py are
assumptions (there is no intervention history in the data to learn them from), so expected values are
illustrative estimates that scale with those assumptions.

Per customer and action:
    match        = how strongly the model's drivers (or customer facts) point at this action  (0..1)
    success_prob = base_save_rate * match * save_rate_scale            (assumed)
    at_stake     = monthly_charge * horizon_months                     (revenue that leaves if they churn)
    exp_saved    = P(churn) * success_prob * at_stake
    net_value    = exp_saved - cost
"""
from . import actions as A

DRIVER_MATCH = (1.0, 0.85, 0.7)      # strength when the matching driver is the #1 / #2 / #3 risk driver
FACT_MATCH = 0.6                     # strength when only a customer fact (not a model driver) justifies it
FALLBACK_MATCH = 0.5                 # general check-in, when nothing else applies


def _matches(feature: str, patterns) -> bool:
    return any(feature.startswith(p) if p.endswith("_") else feature == p for p in patterns)


def build_context(repo, customer_id):
    """Collect everything the engine needs for one customer from either repository (CSV or Postgres).
    Returns None if the customer has no stored prediction."""
    pred = repo.latest_prediction(customer_id)
    if pred is None:
        return None
    prof = repo.profile(customer_id) or {}
    sub = repo.subscription(customer_id) or {}
    use = repo.usage_summary(customer_id) or {}
    sup = repo.support_summary(customer_id) or {}
    txn = repo.transaction_summary(customer_id) or {}
    return {
        "customer_id": customer_id,
        "churn_probability": float(pred["churn_probability"]),
        "risk_tier": pred["risk_tier"],
        "drivers": list(pred["top_features"]),
        "region": prof.get("region"), "plan": prof.get("plan") or sub.get("plan"),
        "signup_channel": prof.get("signup_channel"),
        "monthly_charge": float(sub.get("monthly_charge") or 0.0),
        "tenure_months": sub.get("tenure_months"),
        "contract_type": sub.get("contract_type"),
        "payment_method": sub.get("payment_method"),
        "usage_trend": use.get("trend"),
        "open_tickets": sup.get("open_tickets", 0) or 0,
        "failed_payments_90d": txn.get("failed_payments_90d", 0) or 0,
    }


def rank_actions(ctx, horizon_months=12, save_rate_scale=1.0):
    """All applicable actions for one customer, best net value first."""
    risk_drivers = [(i, d) for i, d in enumerate(ctx["drivers"]) if d.get("direction") == "increases_risk"]
    at_stake = ctx["monthly_charge"] * horizon_months
    out = []
    for act in A.CATALOG:
        if not act.guard(ctx):
            continue
        match, why = 0.0, []
        for rank, d in risk_drivers:                                  # model-driven match
            if _matches(d["feature"], act.drivers):
                strength = DRIVER_MATCH[min(rank, 2)]
                if strength > match:
                    match = strength
                why.append(f"The model lists {A.label(d['feature'])} as risk driver #{rank + 1} "
                           f"(SHAP {d['shap_value']:+.2f}).")
        if act.fact(ctx):                                             # customer-fact match
            text = act.fact_text(ctx)
            if act.id == "general_checkin":
                if match == 0:
                    match = FALLBACK_MATCH
            elif match < FACT_MATCH:
                match = max(match, FACT_MATCH)
            if text and act.id != "general_checkin":
                why.append(f"Customer fact: {text}.")
            elif act.id == "general_checkin" and match == FALLBACK_MATCH:
                why.append("No single cause stands out, so a general check-in is the safe default.")
        if match == 0:
            continue
        success = min(1.0, act.base_save_rate * match * save_rate_scale)
        cost = act.cost_for(ctx)
        exp_saved = ctx["churn_probability"] * success * at_stake
        out.append({"action_id": act.id, "action": act.name, "owner_team": act.owner_team,
                    "channel": act.channel, "description": act.description,
                    "why": why, "match": round(match, 2), "success_probability": round(success, 3),
                    "cost": round(cost, 2), "expected_revenue_saved": round(exp_saved, 2),
                    "net_value": round(exp_saved - cost, 2),
                    "roi": round((exp_saved - cost) / cost, 2) if cost > 0 else None})
    if any(r["action_id"] != "general_checkin" for r in out):          # fallback only when nothing specific applies
        out = [r for r in out if r["action_id"] != "general_checkin"]
    out.sort(key=lambda r: r["net_value"], reverse=True)
    return out


def plan_campaign(contexts, budget, horizon_months=12, save_rate_scale=1.0, max_contacts=None,
                  team_capacity=None, min_net_value=0.0):
    """Choose who to contact and with what, under a budget (greedy by net value per unit of cost).

    Each customer gets at most one action: their highest net-value action, or the next best one if that
    action's team is at capacity or it no longer fits the budget. Customers are served in order of value
    for money (net value per unit cost of their best action) until the budget or contact limit runs out.
    Greedy is a standard heuristic for budgeted selection, not a proven optimum.
    """
    team_capacity = dict(team_capacity or {})
    team_used = {}
    per_customer = []
    for ctx in contexts:
        opts = [r for r in rank_actions(ctx, horizon_months, save_rate_scale)
                if r["net_value"] > min_net_value and r["cost"] > 0]
        if opts:
            per_customer.append((opts[0]["net_value"] / opts[0]["cost"], ctx, opts))
    per_customer.sort(key=lambda t: t[0], reverse=True)       # best value-for-money customers first

    chosen, spent = [], 0.0
    for _, ctx, opts in per_customer:
        if max_contacts is not None and len(chosen) >= max_contacts:
            break
        for r in opts:                                         # customer's best action first, then fallbacks
            team = r["owner_team"]
            if spent + r["cost"] > budget:
                continue
            if team in team_capacity and team_used.get(team, 0) >= team_capacity[team]:
                continue
            spent += r["cost"]
            team_used[team] = team_used.get(team, 0) + 1
            chosen.append({"customer_id": ctx["customer_id"],
                           "churn_probability": round(ctx["churn_probability"], 4),
                           "risk_tier": ctx["risk_tier"], "monthly_charge": ctx["monthly_charge"],
                           **{k: r[k] for k in ("action_id", "action", "owner_team", "channel", "cost",
                                                "success_probability", "expected_revenue_saved", "net_value")},
                           "why": r["why"][0] if r["why"] else ""})
            break

    by_action, by_team = {}, {}
    for c in chosen:
        by_action[c["action"]] = by_action.get(c["action"], 0) + 1
        by_team[c["owner_team"]] = by_team.get(c["owner_team"], 0) + 1
    summary = {"budget": budget, "spent": round(spent, 2), "customers_contacted": len(chosen),
               "customers_considered": len(contexts),
               "expected_revenue_saved": round(sum(c["expected_revenue_saved"] for c in chosen), 2),
               "expected_net_value": round(sum(c["net_value"] for c in chosen), 2),
               "by_action": by_action, "by_team": by_team, "team_capacity_used": team_used}
    return {"summary": summary, "selected": chosen}
