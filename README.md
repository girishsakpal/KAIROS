# Kairos

An Explainable AI-Driven Customer Churn Intelligence Framework for Predictive and
Actionable Retention.

> The right insight, at the right moment.

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

```bash
python3 src/data_generation/generate_synthetic_data.py \
    --seed-csv data/raw/Telco-Customer-Churn.csv \
    --out-dir data/synthetic
```

Load into Postgres (order matters, respects FK dependencies):

```bash
psql -U postgres -d kairos -c "\copy customers FROM 'data/synthetic/customers.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy subscriptions FROM 'data/synthetic/subscriptions.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy usage_logs FROM 'data/synthetic/usage_logs.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy support_tickets FROM 'data/synthetic/support_tickets.csv' CSV HEADER"
psql -U postgres -d kairos -c "\copy transactions FROM 'data/synthetic/transactions.csv' CSV HEADER"
```
