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

# Results of Linear Regression vs XG Boost:

```
{
  "logistic_regression": {
    "roc_auc": 0.8984,
    "pr_auc": 0.8728,
    "brier_score": 0.2498
  },
  "xgboost": {
    "roc_auc": 0.9047,
    "pr_auc": 0.8824,
    "brier_score": 0.1599
  }
}
```

## ROC AUC (Receiver Operating Characteristic - Area Under the Curve)
### What it signifies: It measures how well the model separates the two classes (e.g., distinguishing between "fraud" and "not fraud"). A score of 1.0 is perfect, and 0.5 is random guessing.
### Interpretation: Both models have excellent discriminative power (approx. 90% chance of ranking a positive instance higher than a negative one). XGBoost has a slight edge.

## PR AUC (Precision-Recall Area Under the Curve)
### What it signifies: This evaluates performance specifically on the positive class. It is highly useful if your dataset is imbalanced (e.g., rare diseases or defaults). A higher score means the model finds positive cases accurately without catching too many false alarms.
### Interpretation: Both models perform strongly here, meaning they handle the positive class well. XGBoost again slightly outperforms Logistic Regression.

## Brier Score
### What it signifies: It measures the accuracy of predicted probabilities (calibration). It is the mean squared difference between the predicted probability and the actual outcome. Lower is better, with 0.0 being a perfect score and 0.25 representing random guessing (for a 50/50 balanced dataset).
### Interpretation: This is the biggest differentiator. Logistic Regression’s score (~0.25) suggests its probability estimates are uncalibrated and closer to random guessing. XGBoost (0.1599) is much lower, meaning its predicted probabilities are far more reliable and accurate.