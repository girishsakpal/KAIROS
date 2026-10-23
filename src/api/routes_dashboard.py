"""
Endpoints that power the interactive dashboard (same /api/v1 prefix):

  GET /api/v1/customers          searchable, filterable, sortable list of scored customers + chart aggregates
  GET /api/v1/model/validation   ROC, threshold table, deciles, calibration, global SHAP (held-out test set)
  GET /api/v1/campaign/curve     expected value of the plan across budgets
"""
from pathlib import Path

import pandas as pd
from flask import Blueprint, current_app, jsonify, request

from ..recommend import actions as A
from . import dashboard as D
from .auth import require_role
from .errors import ApiError
from .routes_recommend import _common, _contexts, _number

bp = Blueprint("dashboard", __name__, url_prefix="/api/v1")


def _list(name):
    raw = request.args.get(name)
    return [x for x in raw.split(",") if x] if raw else None


def _best(ctxs, horizon=12, scale=1.0):
    cache = current_app.extensions.setdefault("kairos_best", {})
    key = (horizon, scale)
    if key not in cache:
        cache[key] = {c["customer_id"]: D.best_action(c, horizon, scale) for c in _contexts()}
    return cache[key]


@bp.get("/customers")
@require_role("ops", "exec", "admin")
def customers():
    tiers, regions, contracts, bands = _list("risk_tier"), _list("region"), _list("contract"), _list("tenure_band")
    for t in tiers or []:
        if t not in ("high", "medium", "low"):
            raise ApiError(400, "VALIDATION_ERROR", "risk_tier must be high, medium or low.")
    sort = request.args.get("sort", "probability")
    if sort not in D.SORTS:
        raise ApiError(400, "VALIDATION_ERROR", f"sort must be one of {sorted(D.SORTS)}.")
    order = request.args.get("order", "desc")
    if order not in ("asc", "desc"):
        raise ApiError(400, "VALIDATION_ERROR", "order must be asc or desc.")
    page = _number("page", 1, 1, 100_000, int)
    size = _number("page_size", 25, 1, 200, int)
    all_ctx = _contexts()
    ctxs = D.filter_contexts(all_ctx, request.args.get("q"), tiers, regions, contracts, bands)
    ctxs = sorted(ctxs, key=D.SORTS[sort], reverse=(order == "desc"))
    best = _best(all_ctx)
    chunk = ctxs[(page - 1) * size: page * size]
    return jsonify({"total": len(ctxs), "page": page, "page_size": size,
                    "items": [D.table_row(c, best.get(c["customer_id"])) for c in chunk],
                    "aggregates": D.aggregates(ctxs, best),
                    "filter_options": {"regions": sorted({c.get("region") for c in all_ctx if c.get("region")}),
                                       "contracts": sorted({c.get("contract_type") for c in all_ctx if c.get("contract_type")}),
                                       "tenure_bands": list(D.TENURE_BANDS)}})


@bp.get("/model/validation")
@require_role("ops", "exec", "admin")
def model_validation():
    cache = current_app.extensions
    if "kairos_validation" not in cache:
        cfg = current_app.config
        path = Path(cfg["MODELS_DIR"]) / cfg["PREDICTIONS_FILE"]
        try:
            df = pd.read_csv(path).drop_duplicates("customer_id", keep="last")
            shap_path = Path(cfg["MODELS_DIR"]) / "shap_global_importance.csv"
            shap = []
            if shap_path.exists():
                s = pd.read_csv(shap_path).head(12)
                shap = [{"feature": f, "label": A.neutral(f.replace("num__", "").replace("cat__", "")),
                         "mean_abs_shap": round(float(v), 4)} for f, v in zip(s.feature, s.mean_abs_shap)]
            cache["kairos_validation"] = D.validation_payload(df["actual_churn"], df["churn_probability"], shap=shap,
                                                          tiebreak=df["churn_probability_raw"] if "churn_probability_raw" in df else None)
        except (OSError, KeyError, ValueError) as e:
            raise ApiError(503, "VALIDATION_UNAVAILABLE", f"Held-out validation data is not available: {e}")
    return jsonify(cache["kairos_validation"])


@bp.get("/campaign/curve")
@require_role("ops", "admin")
def campaign_curve():
    horizon, scale = _common()
    max_budget = _number("max_budget", 3000.0, 1, 1_000_000)
    points = _number("points", 12, 2, 40, int)
    return jsonify({"curve": D.budget_curve(_contexts(), max_budget, points, horizon, scale),
                    "assumptions": {"horizon_months": horizon, "save_rate_scale": scale}})
