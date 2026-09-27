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

# Results: Logistic Regression vs XGBoost vs LSTM

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
  },
  "lstm": {
    "roc_auc": 0.9174,
    "pr_auc": 0.8957,
    "brier_score": 0.1414
  },
  "lstm_calibrated": {
    "roc_auc": 0.9158,
    "pr_auc": 0.8851,
    "brier_score": 0.1137
  }
}
```

The LSTM is a hybrid model: an LSTM branch reads each customer's raw 6-month
usage sequence (session count, session minutes, feature adoption), concatenated
with a dense branch reading the same static features LR/XGBoost use (contract,
tenure, payment method, support/transaction aggregates), so it's judged on
whether *learning* the usage trajectory beats *summarizing* it into hand-built
features. `lstm_calibrated` applies isotonic calibration fit on a held-out
validation slice — see the Brier section below for why that matters.

## ROC AUC (Receiver Operating Characteristic - Area Under the Curve)
### What it signifies: 
It measures how well the model separates the two classes (e.g., distinguishing between "fraud" and "not fraud"). A score of 1.0 is perfect, and 0.5 is random guessing.
### Interpretation: 
All three models have strong discriminative power. The LSTM has the edge (0.917 raw / 0.916 calibrated) over XGBoost (0.905) and Logistic Regression (0.898), the raw usage sequence appears to carry more signal than the 6 hand-built summary features derived from it.

## PR AUC (Precision-Recall Area Under the Curve)
### What it signifies: 
This evaluates performance specifically on the positive class. It is highly useful if your dataset is imbalanced (e.g., rare diseases or defaults). A higher score means the model finds positive cases accurately without catching too many false alarms.
### Interpretation: 
The LSTM again leads (0.896 raw), ahead of XGBoost (0.882) and Logistic Regression (0.873). Calibration trades a little PR-AUC for much better calibration (0.885) — expected, since PR-AUC is sensitive to how probabilities are ranked/thresholded, not just their calibration.

## Brier Score
### What it signifies: 
It measures the accuracy of predicted probabilities (calibration). It is the mean squared difference between the predicted probability and the actual outcome. Lower is better, with 0.0 being a perfect score.
### Interpretation: 
The calibrated LSTM is the best-calibrated model by a clear margin (0.114), followed by the raw LSTM (0.141) and XGBoost (0.160). **Caveat on LR and XGBoost's scores above:** both are trained with `class_weight`/`scale_pos_weight` re-balancing, which systematically inflates predicted probabilities (mean predicted churn on the test set is 0.565 vs an actual rate of 0.409) — so their Brier scores here are worse than their true ranking ability would suggest, not evidence that their probabilities are "close to random." The LSTM's isotonic calibration step corrects for this directly (mean predicted drops to 0.421 against the same 0.409 actual rate), which is why it's the most trustworthy of the four for anything using the probability itself (e.g. expected-revenue-at-risk), not just the ranking.

**Model selection:** on ROC-AUC the LSTM currently outperforms XGBoost, so it is the promoted champion in `model_registry_row.json` unless retrained. See `docs/LSTM_Notes.md` for architecture details and `docs/Phase3_Findings.md` for the calibration issue in more depth.