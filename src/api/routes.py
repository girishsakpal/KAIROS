"""
/api/v1 endpoints (Phase 3 skeleton): POST /predict, GET /customer-detail/<id>,
GET /model-metrics. Contract: docs/Kairos_API_Contracts.md.

NOT built yet (will 404): /batch-predict, /segment, /chat.

Deliberate differences from the contract (flagged, not hidden):
  * POST /predict serves the CACHED prediction only. force_recompute=true returns
    501 NOT_IMPLEMENTED (re-scoring needs the champion model loaded in-process).
  * A customer that exists but was never scored gets 404 PREDICTION_NOT_FOUND
    (the contract only lists NOT_FOUND for unknown customers). Only the 1,419
    held-out test customers have predictions right now.
  * drift_status carries an extra `drift_checked` flag: PSI/KS monitoring is Phase 6,
    so `drift_detected: false` currently means "not checked", not "checked and clean".
"""
from datetime import datetime, timezone

from flask import Blueprint, current_app, jsonify, request

from .auth import require_role
from .errors import ApiError

bp = Blueprint("v1", __name__, url_prefix="/api/v1")

INCLUDE_SECTIONS = {
    "subscriptions": ("subscription", "subscription"),
    "usage": ("usage_summary", "usage_summary"),
    "support": ("support_summary", "support_summary"),
    "transactions": ("transaction_summary", "transaction_summary"),
    "prediction": ("latest_prediction", None),
}


def _repo():
    return current_app.extensions["kairos_repo"]


def _now():
    return current_app.config["NOW"] or datetime.now(timezone.utc)


def _drift_status(model):
    trained = datetime.fromisoformat(model["trained_at"])
    if trained.tzinfo is None:
        trained = trained.replace(tzinfo=timezone.utc)
    days = (_now() - trained).days
    return {"is_stale": days > current_app.config["STALE_AFTER_DAYS"],
            "days_since_training": days,
            "drift_detected": bool(model.get("drift_detected", False)),
            "drifted_features": [],
            "drift_checked": model.get("drift_psi_score") is not None}


def _model_block(m):
    if m is None:
        return None
    return {"model_id": m["model_id"], "trained_at": m["trained_at"], "algorithm": m["algorithm"],
            "metrics": {"roc_auc": m.get("roc_auc"), "pr_auc": m.get("pr_auc"),
                        "calibration_error": m.get("calibration_error")},
            "drift": {"psi_score": m.get("drift_psi_score"), "ks_test_p_value": m.get("drift_ks_pvalue"),
                      "drift_detected": bool(m.get("drift_detected", False)), "flagged_features": []},
            "status": m.get("status")}


@bp.get("/health")
def health():
    serving = _repo().serving_model()
    return jsonify({"status": "ok", "serving_model_id": serving["model_id"] if serving else None,
                    **_repo().counts()})


@bp.post("/predict")
@require_role("ops", "admin")
def predict():
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ApiError(400, "VALIDATION_ERROR", "Request body must be a JSON object.")
    cid = body.get("customer_id")
    if not isinstance(cid, str) or not cid.strip():
        raise ApiError(400, "VALIDATION_ERROR", "'customer_id' is required and must be a non-empty string.")
    force = body.get("force_recompute", False)
    if not isinstance(force, bool):
        raise ApiError(400, "VALIDATION_ERROR", "'force_recompute' must be a boolean.")

    repo = _repo()
    if not repo.customer_exists(cid):
        raise ApiError(404, "NOT_FOUND", f"Customer '{cid}' does not exist.")
    model = repo.serving_model()
    if model is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "No serving model is registered.")
    if force:
        raise ApiError(501, "NOT_IMPLEMENTED",
                       "force_recompute is not implemented yet; only cached predictions are served.")
    pred = repo.latest_prediction(cid)
    if pred is None:
        raise ApiError(404, "PREDICTION_NOT_FOUND",
                       f"Customer '{cid}' exists but has no stored prediction.")

    return jsonify({"customer_id": cid, "churn_probability": pred["churn_probability"],
                    "risk_tier": pred["risk_tier"], "model_id": pred["model_id"],
                    "model_trained_at": model["trained_at"], "model_status": model.get("status"),
                    "drift_status": _drift_status(model), "top_features": pred["top_features"],
                    "predicted_at": pred["predicted_at"], "recomputed": False})


@bp.get("/customer-detail/<customer_id>")
@require_role("ops", "admin")
def customer_detail(customer_id):
    repo = _repo()
    raw = request.args.get("include")
    if raw is None:
        wanted = set(INCLUDE_SECTIONS)
    else:
        wanted = {s.strip() for s in raw.split(",") if s.strip()}
        bad = sorted(wanted - set(INCLUDE_SECTIONS))
        if bad or not wanted:
            raise ApiError(400, "VALIDATION_ERROR", f"Unsupported 'include' value(s): {bad or raw!r}.",
                           {"allowed": sorted(INCLUDE_SECTIONS)})
    if not repo.customer_exists(customer_id):
        raise ApiError(404, "NOT_FOUND", f"Customer '{customer_id}' does not exist.")

    out = {"customer_id": customer_id, "profile": repo.profile(customer_id)}
    if "subscriptions" in wanted:
        out["subscription"] = repo.subscription(customer_id)
    if "usage" in wanted:
        out["usage_summary"] = repo.usage_summary(customer_id)
    if "support" in wanted:
        out["support_summary"] = repo.support_summary(customer_id)
    if "transactions" in wanted:
        out["transaction_summary"] = repo.transaction_summary(customer_id)
    if "prediction" in wanted:
        pred, model = repo.latest_prediction(customer_id), repo.serving_model()
        if pred is None or model is None:
            out["latest_prediction"] = None
        else:
            ds = _drift_status(model)
            out["latest_prediction"] = {
                "churn_probability": pred["churn_probability"], "risk_tier": pred["risk_tier"],
                "top_features": [d["feature"] for d in pred["top_features"]],
                "drift_status": {"is_stale": ds["is_stale"], "days_since_training": ds["days_since_training"]}}
    return jsonify(out)


@bp.get("/model-metrics")
@require_role("exec", "admin")
def model_metrics():
    repo = _repo()
    requested_id = request.args.get("model_id")
    requested = None
    if requested_id:
        requested = repo.model_by_id(requested_id)
        if requested is None:
            raise ApiError(404, "NOT_FOUND", f"No model with id '{requested_id}' in the registry.")
    include_history = request.args.get("include_history", "false").lower() == "true"

    body = {"champion_model": _model_block(repo.champion_model()),
            # The model whose predictions /predict serves; differs from the champion while
            # the LSTM (no per-customer SHAP) holds the champion slot.
            "serving_model": _model_block(repo.serving_model()),
            "evaluation_harness": {"text_to_sql_execution_accuracy": None, "ragas_faithfulness": None,
                                   "explanation_faithfulness": None,
                                   "note": "Populated starting Phase 7 (Module B evaluation harness)."},
            "history": []}
    if requested is not None:
        body["requested_model"] = _model_block(requested)
    if include_history:
        body["history"] = [b for b in (body["champion_model"], body["serving_model"]) if b]
    return jsonify(body)
