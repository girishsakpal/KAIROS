# Kairos — Tech Stack Lock & Evaluation Metrics Summary (Phase 2)

**Status:** Draft for guide sign-off — locks scope going into Phase 3 (Core Build).
**Companion docs:** `Kairos_SRS.docx` (Appendix B: Tech Stack Summary; Other Requirements:
Evaluation/Data), `Kairos_API_Contracts.md`, `Kairos_ER_Diagram.pdf`.
**Purpose:** Pull the tech stack and evaluation methodology out of the SRS appendices into
one short, checkable reference — so "what are we building with" and "how do we know it
worked" are each a single page the guide can approve once, instead of scattered across a
10-page document.

---

## 1. Tech Stack Lock

Locking versions/choices now avoids mid-build swaps in Phase 3–6. Anything marked
**[flex]** can still change without re-approval if a genuine blocker comes up; everything
else should go back to the guide if it changes.

| Layer | Locked Choice | Notes |
|---|---|---|
| Database | PostgreSQL + `pgvector` extension | Single governed store for both structured tables and embedded free-text/SHAP-explanation vectors — no separate vector DB. |
| ETL / Data Gen | Python, pandas | Synthetic `usage_logs` / `support_tickets` / `transactions` generated conditioned on churn outcome, not random. |
| Core ML | scikit-learn (baseline Logistic Regression), XGBoost **[flex: or LightGBM]** | Baseline + main model per Objective 2. Pick one gradient-boosted library and stay with it through Phase 3–6 once trained. |
| Tuning | Optuna **[flex: or GridSearch]** | |
| Explainability | SHAP | Global + per-customer local explanations; feeds `churn_predictions.top_features`. |
| Experiment Tracking | MLflow | Also backs `model_registry` / champion-challenger promotion. |
| Drift Detection | SciPy (PSI, KS-test) | Flags feeding into `/model-metrics` and `/predict` drift_status. |
| Embeddings | sentence-transformers | Embeds free-text columns and SHAP-explanation text into `pgvector`. |
| RAG Orchestration | LangChain **[flex: or lightweight custom orchestrator]** | Provider-agnostic LLM layer, `.env`-configurable (OpenAI API or local Ollama). |
| Backend API | Flask, SQLAlchemy | Per `Kairos_API_Contracts.md`. |
| API Docs / Auth / Tests | Flask-RESTX or Flask-Smorest **[flex, pick one]**, Flask-JWT-Extended, Pytest | |
| BI — Operations | Power BI | At-risk list, agent workload, drill-through into `/customer-detail`. |
| BI — Executive | Tableau | Churn trend, cohort retention, revenue-at-risk, aggregated SHAP drivers. |
| Containerization | Docker, docker-compose | |
| CI | GitHub Actions | Runs Pytest + (from Phase 7) the evaluation harness. |
| Evaluation Tooling | Custom text-to-SQL execution-accuracy harness, RAGAS, custom explanation-faithfulness scorer | See Section 2 below. |

**Explicit non-choices (out of scope, don't revisit without a scope conversation):**
- No separate vector database (Pinecone, Weaviate, etc.) — pgvector only.
- No domain-general text-to-SQL beyond the churn schema (already agreed — see Section 3
  scope boundary in the handoff doc).
- No mobile app / native frontend — chat surface is embedded web, alongside Power
  BI/Tableau, not a replacement for either.

---

## 2. Evaluation Metrics Lock

Two tracks: **Core** metrics (needed for every DS interview/report regardless of Module
B), and **Module B** metrics (needed for the paper and full differentiation story, land in
Phase 7).

### 2.1 Core — Model Quality

| Metric | What it measures | When computed |
|---|---|---|
| ROC-AUC | Overall discrimination ability | Phase 3 onward, every training run |
| PR-AUC | Discrimination under class imbalance (churn is imbalanced) | Phase 3 onward |
| Calibration error | Whether predicted probabilities match observed churn rates | Phase 3 onward |
| Expected revenue saved | Business-impact framing — $ retained if top-risk decile is acted on at assumed outreach success rate | Once real data/results exist (Section 5 open item) |

### 2.2 Core — Drift Monitoring

| Metric | What it measures | Threshold / action |
|---|---|---|
| PSI (Population Stability Index) | Feature distribution shift vs. training baseline | Flag `drift_detected=true` above agreed PSI threshold (recommend locking a specific number, e.g. 0.2, with the guide) |
| KS-test p-value | Statistical distribution shift per feature | Flag if p-value below agreed significance threshold |

**Open item:** exact PSI/p-value thresholds are not yet locked — recommend agreeing a
specific number with the guide before Phase 6, since "drift detected" currently reads as
qualitative in the docs.

### 2.3 Module B — Conversational Layer (Phase 7 build target)

| Metric | What it measures | Method |
|---|---|---|
| Text-to-SQL execution accuracy | Whether generated SQL returns the correct result set | Curated question/gold-SQL test set, extended with ambiguous/unanswerable questions (PRACTIQ-style design) |
| RAGAS-style faithfulness | Whether the answer is grounded in retrieved context, not invented | RAGAS framework, applied to semantic/hybrid questions |
| RAGAS-style answer relevance | Whether the answer actually addresses the question asked | RAGAS framework |
| Explanation faithfulness (novel metric) | Whether a "why" answer's stated top drivers match the model's real SHAP `top_features` for that prediction | Custom scorer: overlap/precision between answer-stated drivers and stored ground truth |
| Drift-awareness check | Whether the system correctly surfaces staleness/drift warnings and doesn't overstate confidence when they're active | Scripted test cases against known-stale model states |

### 2.4 Optional — User Study

A short task-based usability comparison (time-to-insight, correctness) for ops/exec users
using the chat interface vs. raw dashboards. Strengthens the paper; not required for the
core DS story or the Sem VII checkpoint.

---

## 3. Sign-off Checklist

Use this as the actual guide-facing artifact — a single page to walk through in a meeting:

- [ ] Tech stack table (Section 1) approved as-is, or **[flex]** items resolved
- [ ] PSI / KS-test drift thresholds agreed on a specific number (currently open)
- [ ] Core evaluation metrics (Section 2.1–2.2) confirmed sufficient for Sem VII checkpoint
- [ ] Module B metrics (Section 2.3) confirmed as Phase 7 scope, not expected earlier
- [ ] ER diagram field-level schema (separate open item, not covered in this doc) reviewed
      before Phase 3 database build begins

---

## 4. Traceability back to source docs

- Tech stack table mirrors `Kairos_Project_Documentation.docx` Section 8, cross-checked
  against `Kairos_SRS.docx` Appendix B.
- Evaluation metrics mirror `Kairos_Project_Documentation.docx` Section 12, cross-checked
  against `Kairos_SRS.docx` Other Requirements (Evaluation/Data) and Objective 7.
- This doc adds no new decisions beyond what's already in those two sources — it exists
  purely to make Phase 2 sign-off faster and more concrete.
