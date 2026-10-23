"""
Retention Action Engine endpoints (same /api/v1 prefix):

  GET /api/v1/recommend/catalog     the action catalogue with its (assumed) costs and save rates
  GET /api/v1/recommend/<id>        ranked retention actions for ONE customer, with reasons
  GET /api/v1/campaign              budget-constrained plan: who to contact and with which action

These are recommendations to act on, not guarantees: save rates and costs are assumptions
(see src/recommend/actions.py), so expected values are illustrative estimates.
"""
import threading

from flask import Blueprint, current_app, jsonify, request

from ..recommend import actions as A
from ..recommend import engine
from .auth import require_role
from .errors import ApiError
from .repository import CsvRepository

_BUILD_LOCK = threading.Lock()
bp = Blueprint("recommend", __name__, url_prefix="/api/v1")

DISCLAIMER = ("Recommendations are decision support, not a guarantee that a customer will stay. "
              "Costs and save rates are assumptions, not measured results.")


def _repo():
    return current_app.extensions["kairos_repo"]


def _number(name, default, lo, hi, cast=float):
    raw = request.args.get(name)
    if raw is None:
        return default
    try:
        val = cast(raw)
    except ValueError:
        raise ApiError(400, "VALIDATION_ERROR", f"'{name}' must be a number.")
    if not lo <= val <= hi:
        raise ApiError(400, "VALIDATION_ERROR", f"'{name}' must be between {lo} and {hi}.")
    return val


def _common():
    return (_number("horizon_months", 12, 1, 60, int), _number("save_rate_scale", 1.0, 0.1, 3.0))


@bp.get("/recommend/catalog")
@require_role("ops", "exec", "admin")
def catalog():
    items = []
    for a in A.CATALOG:
        cost = a.cost if not callable(a.cost) else "scales with the customer's monthly charge"
        items.append({"action_id": a.id, "action": a.name, "owner_team": a.owner_team, "channel": a.channel,
                      "description": a.description, "assumed_save_rate": a.base_save_rate, "cost": cost})
    return jsonify({"actions": items, "disclaimer": DISCLAIMER})


@bp.get("/recommend/<customer_id>")
@require_role("ops", "admin")
def recommend(customer_id):
    horizon, scale = _common()
    limit = _number("limit", 3, 1, 8, int)
    repo = _repo()
    if not repo.customer_exists(customer_id):
        raise ApiError(404, "NOT_FOUND", f"Customer '{customer_id}' does not exist.")
    ctx = engine.build_context(repo, customer_id)
    if ctx is None:
        raise ApiError(404, "PREDICTION_NOT_FOUND", f"Customer '{customer_id}' exists but has no stored prediction.")
    ranked = engine.rank_actions(ctx, horizon, scale)
    return jsonify({
        "customer_id": customer_id, "churn_probability": round(ctx["churn_probability"], 4),
        "risk_tier": ctx["risk_tier"], "monthly_charge": ctx["monthly_charge"],
        "drivers": [{"feature": d["feature"], "label": A.label(d["feature"]), "name": A.neutral(d["feature"]), "shap_value": d["shap_value"],
                     "direction": d["direction"]} for d in ctx["drivers"]],
        "recommendations": ranked[:limit], "total_applicable_actions": len(ranked),
        "assumptions": {"horizon_months": horizon, "save_rate_scale": scale,
                        "revenue_at_stake": round(ctx["monthly_charge"] * horizon, 2)},
        "disclaimer": DISCLAIMER})


def _contexts():
    """Contexts for every scored customer, built once (from the local CSV files, so it is fast even when
    the API itself runs on Postgres; both repositories return identical summaries)."""
    cache = current_app.extensions
    with _BUILD_LOCK:                       # the warm-up thread and a first request must not both build it
        if "kairos_contexts" not in cache:
            repo = _repo()
            if not hasattr(repo, "scored_customer_ids"):
                repo = CsvRepository(current_app.config)
            cache["kairos_contexts"] = [c for c in (engine.build_context(repo, cid)
                                                    for cid in repo.scored_customer_ids()) if c]
        return cache["kairos_contexts"]


@bp.get("/campaign")
@require_role("ops", "admin")
def campaign():
    horizon, scale = _common()
    budget = _number("budget", 500.0, 1, 1_000_000)
    max_contacts = _number("max_contacts", None, 1, 100_000, int) if "max_contacts" in request.args else None
    limit = _number("limit", 50, 1, 2000, int)
    capacity = {}
    for team in {a.owner_team for a in A.CATALOG}:
        key = "capacity_" + team.replace(" ", "_")
        if key in request.args:
            capacity[team] = _number(key, 0, 0, 100_000, int)
    plan = engine.plan_campaign(_contexts(), budget, horizon, scale, max_contacts, capacity)
    sel = plan["selected"]
    return jsonify({"summary": plan["summary"], "selected": sel[:limit], "returned": min(limit, len(sel)),
                    "assumptions": {"horizon_months": horizon, "save_rate_scale": scale,
                                    "team_capacity": capacity or None},
                    "disclaimer": DISCLAIMER})
