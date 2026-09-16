# Kairos

An Explainable AI-Driven Customer Churn Intelligence Framework for Predictive and
Actionable Retention.

> The right insight, at the right moment.

Formal project title (use on SRS/reports/paper): *"An Explainable AI-Driven Customer
Churn Intelligence Framework for Predictive and Actionable Retention."* Codename
"Kairos" is used everywhere else (repo, resume, viva, demo).

---

## Repo structure

```
kairos/
├── data/
│   ├── raw/            # place the real Kaggle Telco-Customer-Churn.csv here (gitignored)
│   ├── synthetic/       # generator output — customers/subscriptions/usage/tickets/txns
│   └── processed/       # feature-engineered, leakage-safe train/test sets (Phase 3, next step)
├── db/
│   └── schema.sql        # PostgreSQL DDL — 7 tables, matches Kairos_ER_Diagram.pdf
├── src/
│   ├── data_generation/  # generate_synthetic_data.py — conditioned on churn outcome
│   ├── features/         # leakage-safe feature engineering (Phase 3, next step)
│   ├── models/            # baseline LR + XGBoost/LightGBM training, SHAP, MLflow (Phase 3)
│   ├── api/                # Flask + SQLAlchemy REST API — see docs/Kairos_API_Contracts.md
│   └── evaluation/         # text-to-SQL accuracy, RAGAS, explanation-faithfulness (Phase 7)
├── dashboards/
│   ├── powerbi/            # .pbix files + JSON theme file (Phase 3 ops view, Phase 6 full)
│   └── tableau/            # .twbx files + .tps theme file (Phase 6 exec view)
├── docs/                    # living reference docs (see below)
├── notebooks/                # exploratory analysis, not production code
├── tests/                     # Pytest suite for src/api and src/models
├── mlruns/                     # MLflow tracking store (gitignored — local/artifact only)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Docs index (`docs/`)

| File | What it is |
|---|---|
| `KAIROS_documentation.pdf` | Living master reference — problem statement, novelty, architecture, timeline, publication plan, resume bullets, verified reference list |
| `Kairos_Handoff_11Sep2026.md` | Session handoff / running project journal |
| `Kairos_API_Contracts.md` | REST endpoint contracts for the 6 API routes |
| `Kairos_TechStack_Evaluation_Lock.md` | Locked tech stack + evaluation metrics, for guide sign-off |

Not yet in this repo (produced in earlier sessions, re-attach if needed):
`Kairos_SRS.docx`, `Kairos_ER_Diagram.pdf`/`.png`, `Kairos_Phase1_Feedback_Survey.md`,
`Kairos_Block_Diagram_Explanation.pdf`, `Kairos_Diagram_Explanation.pdf`.

## Setup

```bash
cp .env.example .env          # fill in DB creds / LLM provider config
pip install -r requirements.txt
```

Stand up the database:

```bash
psql -U postgres -c "CREATE DATABASE kairos;"
psql -U postgres -d kairos -f db/schema.sql
```

Generate synthetic data (fallback mode works with no seed file; swap in the real
Kaggle Telco dataset with `--seed-csv` once downloaded):

```bash
python3 src/data_generation/generate_synthetic_data.py \
    --seed-csv data/raw/Telco-Customer-Churn.csv \
    --out-dir data/synthetic
```

Load into Postgres (order matters — respects FK dependencies):

```bash
psql -U postgres -d kairos -c "\copy customers FROM 'data/synthetic/customers.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy subscriptions FROM 'data/synthetic/subscriptions.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy usage_logs FROM 'data/synthetic/usage_logs.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy support_tickets FROM 'data/synthetic/support_tickets.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy transactions FROM 'data/synthetic/transactions.csv' CSV HEADER"
```

## Where things stand (Phase 2 → Phase 3)

Phase 2 (SRS, ER diagram, API contracts, tech stack lock) is done. Phase 3 is in
progress: schema + synthetic data generation are done and verified (churners show
real signal — lower usage, more failed payments, more tickets — see generator output).
Next: leakage-safe feature engineering, then baseline model training with SHAP.

See `docs/Kairos_Handoff_11Sep2026.md` for the full running log.
