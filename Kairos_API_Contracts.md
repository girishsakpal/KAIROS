# Kairos — API Contracts (Phase 2)

**Status:** Draft for guide review. Companion to `Kairos_SRS.docx` (Section: External
Interfaces) and `Kairos_ER_Diagram.pdf`.
**Framework:** Flask + SQLAlchemy, documented via Flask-RESTX / Flask-Smorest (OpenAPI),
auth via Flask-JWT-Extended.
**Base path:** `/api/v1`
**Auth:** All endpoints require a valid JWT (`Authorization: Bearer <token>`) unless
noted. Roles: `ops` (Power BI / support agent access), `exec` (Tableau / leadership
access), `admin` (full access, model-metrics + model_registry writes).

Error format (shared across all endpoints):

```json
{
  "error": {
    "code": "string (e.g. NOT_FOUND, VALIDATION_ERROR, MODEL_UNAVAILABLE)",
    "message": "human-readable description",
    "details": {}
  }
}
```

Common status codes: `200 OK`, `201 Created`, `400 Bad Request`, `401 Unauthorized`,
`403 Forbidden`, `404 Not Found`, `422 Unprocessable Entity`, `500 Internal Server Error`,
`503 Service Unavailable` (model registry has no active champion model).

---

## 1. `POST /predict` — Single-customer churn prediction

**Maps to:** FR-2.1 (Prediction & Explainability) · **Priority:** High · **Layer:** Core

Runs (or retrieves a cached) churn prediction for one customer and returns the score plus
top SHAP drivers.

### Request

```json
{
  "customer_id": "string (required)",
  "force_recompute": false
}
```

- `force_recompute` (bool, optional, default `false`): if `true`, bypasses any cached
  `churn_predictions` row and re-scores using the current champion model.

### Response `200 OK`

```json
{
  "customer_id": "CUST-004471",
  "churn_probability": 0.82,
  "risk_tier": "high",
  "model_id": "xgb_churn_v7",
  "model_trained_at": "2026-07-24T00:00:00Z",
  "drift_status": {
    "is_stale": true,
    "days_since_training": 41,
    "drift_detected": true,
    "drifted_features": ["usage_logs.session_count_30d"]
  },
  "top_features": [
    {"feature": "failed_payments_90d", "shap_value": 0.41, "direction": "increases_risk"},
    {"feature": "tenure_months", "shap_value": -0.18, "direction": "decreases_risk"},
    {"feature": "support_tickets_60d", "shap_value": 0.15, "direction": "increases_risk"}
  ],
  "predicted_at": "2026-09-11T10:03:00Z"
}
```

### Errors
- `404 NOT_FOUND` — customer_id doesn't exist in `customers`.
- `503 MODEL_UNAVAILABLE` — no champion model registered in `model_registry`.

### Notes
- `drift_status` is what makes this endpoint drift-aware per your novelty framing (Section
  6.2 of the master doc) — every prediction response carries staleness/confidence context,
  not just a bare probability.
- `top_features` is sourced from (or written to) `churn_predictions.top_features` JSONB.

---

## 2. `POST /batch-predict` — Bulk churn scoring

**Maps to:** FR-2.2 · **Priority:** High · **Layer:** Core

Scores a list or filtered segment of customers in one call — used by the nightly scoring
job and by the ops dashboard's "refresh at-risk list" action.

### Request

```json
{
  "customer_ids": ["CUST-001", "CUST-002"],
  "filter": {
    "region": "West",
    "subscription_status": "active"
  },
  "async": true
}
```

- Either `customer_ids` **or** `filter` is provided, not both. If neither is given,
  scores the entire active customer base (admin-only).
- `async` (bool, default `true` for >500 customers): if `true`, returns a job handle
  immediately; if `false`, blocks and returns full results (capped at 500 rows).

### Response `202 Accepted` (async)

```json
{
  "job_id": "batch-2026-09-11-0007",
  "status": "queued",
  "customer_count": 1840,
  "status_url": "/api/v1/batch-predict/batch-2026-09-11-0007"
}
```

### Response `200 OK` (sync, small batch)

```json
{
  "results": [
    {"customer_id": "CUST-001", "churn_probability": 0.12, "risk_tier": "low"},
    {"customer_id": "CUST-002", "churn_probability": 0.77, "risk_tier": "high"}
  ],
  "model_id": "xgb_churn_v7",
  "scored_at": "2026-09-11T10:05:00Z"
}
```

### `GET /batch-predict/{job_id}` — poll job status

Returns `queued` / `running` / `completed` / `failed`, plus a results download link when
`completed`.

### Errors
- `400 VALIDATION_ERROR` — both or neither of `customer_ids`/`filter` provided.
- `403 FORBIDDEN` — non-admin requesting full active-base scoring.

---

## 3. `GET /customer-detail/{customer_id}` — 360° customer view

**Maps to:** FR-2.3 · **Priority:** High · **Layer:** Core

Powers drill-through from both dashboards and the chat layer's "tell me about customer X"
queries. Aggregates across all seven tables.

### Request

Path param: `customer_id`. Optional query params:
`?include=subscriptions,usage,support,transactions,prediction` (default: all).

### Response `200 OK`

```json
{
  "customer_id": "CUST-004471",
  "profile": {
    "signup_date": "2025-02-14",
    "region": "West",
    "plan": "Pro Monthly"
  },
  "subscription": {"status": "active", "monthly_charge": 49.99, "tenure_months": 19},
  "usage_summary": {"sessions_30d": 4, "sessions_90d_avg": 11, "trend": "declining"},
  "support_summary": {"open_tickets": 1, "tickets_60d": 2, "last_ticket_category": "billing"},
  "transaction_summary": {"failed_payments_90d": 2, "lifetime_value": 949.81},
  "latest_prediction": {
    "churn_probability": 0.82,
    "risk_tier": "high",
    "top_features": ["failed_payments_90d", "tenure_months", "support_tickets_60d"],
    "drift_status": {"is_stale": true, "days_since_training": 41}
  }
}
```

### Errors
- `404 NOT_FOUND` — unknown `customer_id`.

---

## 4. `GET /segment` — Cohort / segment analytics

**Maps to:** FR-2.4 · **Priority:** Medium · **Layer:** Core

Returns aggregated churn stats for a cohort, used for the Section 4.1-style "cohort
insight" and "segment recommendation" statements on both dashboards.

### Request (query params)

```
?group_by=signup_channel,plan_type
&filter=region:West,support_tickets_60d:>=2
&metric=churn_rate,avg_revenue_at_risk
&period=90d
```

- `group_by`: comma-separated column list (from `customers`/`subscriptions`) to define
  the cohort.
- `filter`: comma-separated `field:op:value` predicates.
- `metric`: which aggregate metrics to compute.
- `period`: rolling window for time-based metrics (e.g. churn rate over last N days).

### Response `200 OK`

```json
{
  "cohorts": [
    {
      "group": {"signup_channel": "paid_search", "plan_type": "Pro"},
      "customer_count": 214,
      "churn_rate": 0.31,
      "base_churn_rate": 0.14,
      "revenue_at_risk": 18240.50,
      "recommendation": "2+ support tickets in first 60 days correlates with ~2.2x base churn rate for this cohort."
    }
  ],
  "computed_at": "2026-09-11T10:10:00Z"
}
```

### Errors
- `400 VALIDATION_ERROR` — invalid column in `group_by` or `filter`, or unsupported
  `metric`.

---

## 5. `GET /model-metrics` — Model & evaluation transparency

**Maps to:** FR-2.5 (drift/MLOps) and FR evaluation requirements · **Priority:** Medium
(High for admin/audit use) · **Layer:** Core, with Module B metrics attached once Phase 7
lands

Exposes current champion model health, drift status, and (once built) the Module B
evaluation harness scores — this is the admin/audit view into `model_registry`.

### Request (query params, all optional)

```
?model_id=xgb_churn_v7
&include_history=false
```

### Response `200 OK`

```json
{
  "champion_model": {
    "model_id": "xgb_churn_v7",
    "trained_at": "2026-07-24T00:00:00Z",
    "algorithm": "XGBoost",
    "metrics": {"roc_auc": 0.91, "pr_auc": 0.74, "calibration_error": 0.03},
    "drift": {
      "psi_score": 0.28,
      "ks_test_p_value": 0.01,
      "drift_detected": true,
      "flagged_features": ["usage_logs.session_count_30d"]
    },
    "status": "active_needs_review"
  },
  "evaluation_harness": {
    "text_to_sql_execution_accuracy": null,
    "ragas_faithfulness": null,
    "explanation_faithfulness": null,
    "note": "Populated starting Phase 7 (Module B evaluation harness)."
  },
  "history": []
}
```

- `evaluation_harness` fields are `null`/omitted until Phase 7; keep the field names
  stable now so the SRS contract doesn't change later.

### Errors
- `404 NOT_FOUND` — no model with given `model_id` in `model_registry`.

---

## 6. `POST /chat` (a.k.a. `/query`) — Conversational retrieval

**Maps to:** FR-2.6 (Conversational Retrieval) · **Priority:** High for Module B ·
**Layer:** Module B

The natural-language entry point. Classifies the question, routes to SQL / semantic /
hybrid retrieval, and returns a grounded, drift-aware answer with full auditability.

### Request

```json
{
  "question": "Why is customer 4471 flagged high risk, and what should I do?",
  "session_id": "chat-sess-88f2",
  "context_customer_id": "CUST-004471"
}
```

- `session_id` (optional): enables multi-turn follow-ups within the same conversation.
- `context_customer_id` (optional): pins the query to a specific customer, e.g. when
  launched from the customer-detail drill-through in a dashboard.

### Response `200 OK`

```json
{
  "answer": "Customer 4471 is flagged high risk (82% probability) primarily due to 2 failed payments in the last 90 days and 2 support tickets in the last 60 days, offset slightly by 19 months of tenure. Note: the model was trained 41 days ago and drift has been detected in recent usage patterns, so treat this score with some caution. Recommended action: proactive outreach, since usage has been declining for this customer.",
  "route_taken": "hybrid",
  "retrieved_context": {
    "sql_executed": "SELECT * FROM churn_predictions WHERE customer_id = 'CUST-004471' ORDER BY predicted_at DESC LIMIT 1;",
    "semantic_matches": [
      {"source": "support_tickets", "ticket_id": "TCK-9931", "similarity": 0.87}
    ],
    "explanation_object_id": "shap-exp-004471-v7"
  },
  "drift_warning": {
    "is_stale": true,
    "days_since_training": 41,
    "drift_detected": true
  },
  "confidence": "medium",
  "session_id": "chat-sess-88f2"
}
```

### Errors
- `422 VALIDATION_ERROR` — question could not be classified into any supported route
  (aggregation / lookup / semantic / hybrid).
- `503 MODEL_UNAVAILABLE` — underlying churn model unavailable for explanation-grounded
  questions.

### Notes
- `retrieved_context` is what makes answers auditable per your "why this isn't just
  ChatGPT with a database" talking point — the actual SQL and retrieved objects are always
  returned alongside the answer, never hidden.
- `route_taken` values (`sql`, `semantic`, `hybrid`) should be logged for the Phase 7
  evaluation harness (text-to-SQL execution accuracy is measured per route).

---

## Cross-cutting notes for the SRS

- **Auth mapping:** `ops` role → `/predict`, `/batch-predict`, `/customer-detail`,
  `/segment`, `/chat`. `exec` role → `/segment`, `/chat`, read-only `/model-metrics`.
  `admin` role → all endpoints, including full-base `/batch-predict` and
  `/model-metrics` writes (model promotion, not exposed here — internal MLflow/CI
  concern, not a public contract).
- **Versioning:** `/api/v1` prefix now; bump to `/v2` only on a breaking change (e.g. if
  the explanation-faithfulness field shape changes after Phase 7 results come in).
- **Rate limiting:** not yet specified — flag as an open NFR item if your SRS's
  Non-Functional Requirements section doesn't already cover it.
- **Open item carried from the handoff doc:** the ER diagram's field-level schema was
  designed without guide review — if any field names above (e.g.
  `failed_payments_90d`, `top_features`) get renamed after review, this doc needs a
  matching pass.
