# Kairos — Project Handoff / Context Doc (Updated)

Use this to bring a new chat up to speed. Paste this whole file as the first message.
Attach `Kairos_Project_Documentation.docx` (full reference) and `Kairos_SRS.docx`
(Phase 2 requirements doc) if you want the assistant to read complete detail.

This is an updated handoff, written after a working session that took the project from
"Phase 0/1 just starting" to "Phase 2 in progress, SRS and ER diagram done."

---

## 1. Who I am / situation

- 7th semester (B.E./B.Tech), final year major project continuing into 8th semester.
- Timeline: August 2026 – October 2026 (Sem VII), then January 2027 – March 2027 (Sem VIII).
- Sem VII checkpoint (~early Oct 2026): literature review + working prototype.
- Sem VIII final submission (~end Mar 2027): full report, deployed system.
- Targeting a classic Data Scientist job (stats/insights/dashboards), so the project
  reads as a DS portfolio piece first, AI/LLM component as a differentiator, not the
  headline.
- Want to publish in a Scopus-indexed or IEEE venue if realistically possible.

## 2. What the project is

**Formal project title:**
> "An Explainable AI-Driven Customer Churn Intelligence Framework for Predictive and
> Actionable Retention"

**Codename (use everywhere else):** Kairos
**Backronym:** Knowledge-Augmented Insight & Retention Optimization System
**Tagline:** "The right insight, at the right moment."

**One-line description:** An explainable, conversational customer churn intelligence
platform. Predicts which customers are likely to churn, explains why in plain business
terms, and lets people ask follow-up questions in natural language instead of reading a
dashboard or writing SQL.

**Two layers:**
1. **Core (protect this, build it deeply):** governed PostgreSQL data model (customers,
   subscriptions, usage_logs, support_tickets, transactions, churn_predictions,
   model_registry) → leakage-safe feature engineering → interpretable churn model
   (XGBoost/LightGBM + SHAP) → MLflow tracking + drift monitoring → business insights via
   two dashboards (Power BI for ops, Tableau for execs).
2. **Module B (bonus/differentiator, can flex if time is tight):** hybrid SQL + semantic
   (pgvector) retrieval engine with a query router, so people can ask questions in plain
   English and get answers grounded in real data *and* the model's actual SHAP
   explanations. Includes drift/confidence-awareness in answers and a formal evaluation
   harness (text-to-SQL accuracy, RAGAS-style faithfulness, and an explanation-faithfulness
   metric).

**Positioning rule:** always lead with Core/DS story in interviews, reports, and viva.
Module B is the closing "...and I additionally built..." line, not the headline.

## 3. Research gap framing (refined this session — use this, not older drafts)

Two comparison papers were reviewed in depth this session:
- *Converting Natural Language Query to SQL Query* (Rath, 2020, IJEAST) — a rule-based,
  pre-LLM NL-to-SQL system, tested on tiny databases, no ML/explainability component.
- *A data-driven approach with explainable AI for customer churn prediction in the
  telecommunications industry* (Asif, Arif & Mukheimer, 2025, Results in Engineering) —
  already reference #1 in the Section 15 literature list. Proposes XAI-Churn TriBoost
  (XGBoost + CatBoost + LightGBM soft-voting ensemble + SHAP/LIME), strong performance
  (96.44% accuracy), but explanations exist only as static notebook plots, no chatbot, no
  drift-aware serving, no retrieval.

**Guide-facing framing that was explicitly approved (short, no em dashes, no "combination
of 2 papers" language, doesn't foreground "new metric" as the headline):**

> Explainable churn models can predict who's likely to leave and show why, but that
> explanation only shows up as a chart a data scientist has to open. Chatbots that answer
> questions in plain English exist too, but they only fetch raw data, they don't understand
> a prediction or explain it. Kairos connects the two. A manager can ask "why is this
> customer at risk?" in plain English, and the system answers using the model's real SHAP
> reasoning for that specific customer, not a guess. It also tracks whether the underlying
> model is outdated, so it can warn the user when a prediction shouldn't be fully trusted.
> We're also building a way to automatically check that the chatbot's explanation actually
> matches what the model really found, instead of just sounding convincing.

**Important nuance to carry forward:** the explanation-faithfulness metric is still a real
part of the technical evaluation plan (Section 6 of the SRS). It just should not be
foregrounded as "the novelty" when talking to the guide informally — the gap itself
(nobody connects explainable churn models to a conversational layer with drift-awareness)
is the lead, not the metric.

**Honesty framing on novelty, also already agreed with the guide-facing tone:** Kairos is
not claiming a new ML algorithm or a new retrieval algorithm — both are established
techniques. The contribution is architectural/systems-level: explanations as retrievable
objects, drift/confidence-aware answers, and the explanation-faithfulness metric as one
genuinely new evaluation artifact.

**Scope boundary agreed on "complex queries":** Module B should handle more complex query
*patterns* within the churn schema (multi-table joins, cohort comparisons, trend/time
questions, hybrid SQL+semantic queries) — not expand into a domain-general text-to-SQL
system across arbitrary schemas. That would dilute the DS-first positioning and blow up
scope.

**Why this isn't "just ChatGPT with a database" (useful talking point):** ChatGPT reasons
over whatever fits its context window and is known to be unreliable at counting/summing/
joins in free text. Kairos executes real SQL against the real governed database via
SQLAlchemy, so arithmetic and joins are always exact, not guessed. It also shows the
underlying query/retrieved context for auditability, and stays inside one governed,
reusable database instead of one-off pasted data in a chat session.

## 4. Where things stand (updated this session)

**Documentation:**
- `Kairos_Project_Documentation.docx` — living master reference (naming, exec summary,
  problem statement, DS-positioning, literature review direction, novelty, architecture,
  tech stack, Phase 0–9 timeline, publication plan, evaluation methodology, resume bullets,
  18-paper verified reference list).
- 10-slide pitch deck content (text only, for Gamma), DS-first framing.

**Phase 1 (practitioner validation) — survey built, not yet distributed/collected:**
- `Kairos_Phase1_Feedback_Survey.md` — condensed Google Form draft, 10 required questions
  + 1 optional follow-up contact field, ~2-3 min to complete. 4 sections: About you, Data
  access pain points, Customer churn specifics, Reaction to proposed solution.
- Agreed framing: this is a small informal needs-assessment / convenience sample, useful
  for the SRS problem statement and requirements elicitation, NOT a substitute for or
  addition to the literature review, and should not be described with formal statistical
  language (mean, significance) given the likely sample size (~15-20 responses).
- Form title options and section titles drafted (formal and casual versions).
- Thank-you/confirmation message drafted, with instructions for adding a clickable
  LinkedIn link (Settings → Presentation → Confirmation message, paste full profile URL).
- Distribution messages drafted: WhatsApp direct message, WhatsApp status, and two
  different LinkedIn post versions (one architecture-led, one cost-of-churn-led).
- **Not yet done:** actually building the Google Form from the draft, distributing it,
  collecting responses.

**Phase 2 (Requirements & Design) — in progress:**
- `Kairos_SRS.docx` — DONE. 10-page IEEE-830-style SRS: Introduction, Overall Description,
  Functional Requirements (grouped into 6 feature areas: data ingestion, prediction/
  explainability, drift/MLOps, dashboards, conversational retrieval, REST API — each
  requirement has an ID like FR-x.x and a priority of High/Medium mapped to Core/Module B),
  External Interfaces, Non-Functional Requirements, Other Requirements (evaluation/data),
  Appendix A (Glossary), Appendix B (Tech Stack Summary).
- `Kairos_ER_Diagram.pdf` and `Kairos_ER_Diagram.png` — DONE. Full entity-relationship
  diagram for all 7 tables (customers as central entity; subscriptions, usage_logs,
  support_tickets, transactions, churn_predictions as one-to-many children; model_registry
  linked one-to-many into churn_predictions). Color-coded by role: teal = core entity,
  navy = behavioral tables, purple = ML prediction output, gray = model tracking/infra.
  Field-level schema (columns, PK/FK) was designed this session since the source doc only
  named tables, not columns — may need adjustment if the guide/plan wants different fields.
- **Not yet done:** API contracts for the 6 REST endpoints (/predict, /batch-predict,
  /customer-detail, /segment, /model-metrics, /chat or /query), and a standalone
  tech-stack-lock + evaluation-metrics summary doc (though both are already partially
  covered inside the SRS appendices).

**Visual design decisions made this session (not yet formally documented anywhere except
here):**
- Color theme locked: teal (#0F6E56-ish) = Core/trust, gray = shared structure/infra,
  purple = Module B/conversational layer (this mirrors the ER diagram and block diagram
  color coding already in use — keep this consistent everywhere going forward).
- Semantic risk colors for dashboards: green = low risk, amber = medium risk, red = high
  risk, with the explicit note to NOT rely on color alone (colorblind accessibility) —
  pair with icons/shapes or always show the risk label as text too.
- Kairos brand idea floated: an hourglass / clock-hand / single-dot-on-timeline mark tied
  to the Greek meaning of "kairos" (the right/critical moment, distinct from chronological
  time), not yet built.
- Noted as a real, non-trivial task for Phase 6: matching the teal/gray/purple palette
  across Power BI and Tableau's own theme systems (Power BI JSON theme files, Tableau
  .tps preference files) — budget actual time for this, it's commonly underestimated.

**Diagrams produced this session:**
- An early simple 9-box layered block diagram (data → DB → ML/retrieval split → answer
  synthesis → API → 3 presentation surfaces), delivered as `Kairos_Block_Diagram_
  Explanation.pdf`, plain-language, no em dashes, no jargon per explicit request.
- User's own more detailed 10-block diagram (uploaded as an image) was explained block by
  block in `Kairos_Diagram_Explanation.pdf`: Data Sources → Ingestion & Governance →
  PostgreSQL Data Model → Feature Engineering → Churn Prediction (+ Drift Monitoring side
  loop) → SHAP Explainability → Action Recommendation → Power BI / Tableau → Conversational
  AI/RAG. This is likely the more authoritative/current diagram going forward.

## 5. Known open items / things to revisit

- [ ] Pick and lock an actual target conference (Springer/IEEE) with a CFP deadline before
      the Sem VII checkpoint — still just "check current CFPs closer to submission time,"
      not locked. (Carried over from before this session, still unresolved.)
- [ ] Confirm live Scopus/IEEE Xplore indexing status of chosen venue right before
      submitting.
- [ ] Once real data/results exist (Phase 3 onward), fill in "4.1 Insight examples" with
      actual numbers instead of placeholders.
- [ ] Build the actual Google Form from `Kairos_Phase1_Feedback_Survey.md`, distribute via
      LinkedIn/WhatsApp/subreddits, collect 15-20+ responses, and fold results into the SRS
      problem statement / requirements section as informal supporting evidence.
- [ ] Write API contracts for the 6 REST endpoints (next planned step, was about to start
      when this session ended).
- [ ] Consider whether the ER diagram's specific field-level schema (designed this session,
      not sourced from the original docs) needs guide review/adjustment before being
      treated as final for Phase 3 database implementation.
- [ ] Design/build an actual Kairos logo mark if time allows (hourglass/clock-hand idea).
- [ ] No mention of prior/source project names anywhere in documentation — Kairos is
      presented as a standalone project from the start (unchanged rule from before).

## 6. Files produced this session (all in outputs, may need to be re-attached to a new chat)

- `Kairos_Block_Diagram_Explanation.pdf` — early simple block diagram + explanation
- `Kairos_Phase1_Feedback_Survey.md` — condensed 10-question Google Form draft
- `Kairos_Diagram_Explanation.pdf` — explanation of the user's own 10-block diagram
- `Kairos_SRS.docx` — full Phase 2 Software Requirements Specification (10 pages)
- `Kairos_ER_Diagram.pdf` / `Kairos_ER_Diagram.png` — entity-relationship diagram (7 tables)

## 7. How to use this in a new chat

Paste this file's content as your first message and say something like: *"This is my
project context from a previous chat, please read it and continue helping me with
[whatever's next, e.g. 'writing the API contracts for Phase 2', 'starting Phase 3 database
build', 'building the actual Google Form']."* Attach `Kairos_Project_Documentation.docx`
and `Kairos_SRS.docx` if you want deep detail, and re-attach any of the files listed in
Section 6 above that are relevant to what you're doing next.
