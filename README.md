# KAIROS

**Knowledge-Augmented Insight & Retention Optimization System**

KAIROS is an AI-based customer churn prediction and retention system. It predicts which subscription customers may leave, explains the main factors behind each prediction, and recommends actions a retention team can take.

**Project:** B.E. Artificial Intelligence and Data Science, Rizvi College of Engineering, University of Mumbai  
**Academic year:** 2026–27  
**Team:** Girish Dayaghan Sakpal, Aryan Prakash Pilwalkar, Kaif Reshamwala  
**Guide:** Prof. Junaid Mandviwala  
**Status:** Semester VII checkpoint

## 1. What KAIROS does

KAIROS helps a business answer three questions:

1. **Who might leave?** A machine-learning model estimates each customer's churn probability.
2. **Why are they at risk?** SHAP explains the main factors behind each prediction.
3. **What should the team do?** A retention engine recommends actions and helps prioritize customers within a budget.

The system includes a Flask REST API, a web dashboard, and a Power BI report.

## 2. Main features

- Customer churn prediction using Logistic Regression, XGBoost, and LSTM.
- Customer-level explanations using SHAP.
- Risk categories: High, Medium, and Low.
- Recommended retention actions, such as payment recovery calls and onboarding calls.
- Budget planner to prioritize customer contacts.
- Interactive dashboard with customer search, filters, charts, and a work queue.
- Power BI report for operational analysis.
- REST API with 10 endpoints.
- **109 automated tests passing.**

## 3. Current implementation status

**Built in Semester VII**
- PostgreSQL database design with 8 tables and a validated data loader.
- Feature engineering and leakage checks.
- Three machine-learning approaches and model evaluation.
- SHAP explanations and retention recommendations.
- Flask API, interactive web app, and four-page Power BI report.
- Basic MLflow experiment logging.

**Planned for Semester VIII**
- Login and JWT authentication.
- Live predictions for new customers.
- Data/model drift monitoring and model registry.
- Conversational interface with SQL and semantic search.
- Tableau executive dashboard.
- Docker, CI, and a model evaluation framework.
- Testing retention actions using real intervention data.

## 4. Technology stack

- **Language:** Python
- **Machine learning:** scikit-learn, XGBoost, TensorFlow/Keras, SHAP
- **Backend:** Flask REST API
- **Database:** PostgreSQL / Supabase, with CSV support
- **Frontend:** HTML, CSS, JavaScript, SVG charts
- **BI:** Power BI
- **Experiment tracking:** MLflow

## 5. Data used

The project starts with a real Telco Customer Churn dataset containing **7,043 customers**. Additional usage logs, support tickets, and transaction records are generated for the project.

Important limitation: the behavioural records are generated using the churn label. This can make model results look better than they would on real-world behavioural data. These results should therefore be treated as a demonstration, not as proof of real-world performance.

## 6. Model results

Results on the held-out test set of **1,419 customers**:

| Metric | Logistic Regression | XGBoost (used by app) | LSTM |
|---|---:|---:|---:|
| Accuracy | 82.2% | 81.5% | 82.5% |
| Precision | 77.8% | 75.2% | 76.0% |
| Recall | 79.0% | 81.8% | 83.5% |
| F1 score | 0.784 | 0.783 | 0.796 |
| ROC-AUC | 0.901 | 0.894 | 0.916 |

The LSTM has the strongest overall metrics in this test, but the app uses XGBoost because it supports the per-customer SHAP explanations used by the recommendation system.

These results come from one test split, and the synthetic behavioural data may make them optimistic.

## 7. Retention Action Engine

The engine uses a customer's risk score, SHAP drivers, and customer details to rank possible actions. Examples include:

- Recover a failed payment and switch the customer to auto-pay.
- Offer a discounted annual contract.
- Make an early-life onboarding call.
- Provide a product-feature walkthrough.
- Follow up on unresolved support issues.

It can plan customer contacts under a budget and team-capacity limits. Action costs and success rates are assumptions because the project does not yet have real data showing which retention actions worked.

## 8. Application pages

The web app includes:

- **Overview:** overall churn risk, charts, filters, and customer list.
- **Customer 360:** individual customer risk, explanations, and recommendations.
- **Retention planner:** budget and contact planning.
- **Work queue:** track selected actions and export them to CSV.
- **Model validation:** examine thresholds and model performance.
- **API explorer:** try the available API endpoints.
- **Power BI report:** view report screenshots or a published report.
- **Project status:** see what is built and what remains.

## 9. Run the application

You need Python and the packages listed in `requirements.txt`.

From the repository root, run:

```powershell
pip install -r requirements.txt
$env:KAIROS_REPOSITORY="csv"
python -m src.api.app
```

Then open **http://127.0.0.1:5000/** in your browser.

The CSV backend is recommended for demos and does not require a database connection. PostgreSQL/Supabase is optional.

## 10. Run the tests

```bash
python -m unittest tests.test_dashboard tests.test_recommend tests.test_api tests.test_api_postgres tests.test_db_loader
```

The project currently has **109 passing automated tests**.

## 11. Key limitations

- Behavioural data is synthetic and generated using the churn label.
- Model evaluation uses one test split; cross-validation and confidence intervals are not yet available.
- Retention-action costs and success rates are assumptions.
- The recommendations are decision support, not a guarantee that a customer will stay.
- Authentication, live scoring, drift monitoring, and the conversational interface are not yet implemented.

## 12. Semester VIII plan

The next phase will focus on authentication, live scoring, monitoring, a conversational interface, Tableau, Docker/CI, improved evaluation, testing retention actions with real data, and the final report and paper.

## 13. Repository overview

```text
kairos/
├── data/          Raw, generated, and processed data
├── db/            Database schema and migrations
├── src/
│   ├── data_generation/
│   ├── features/
│   ├── models/
│   ├── db/
│   ├── recommend/
│   ├── api/
│   ├── ui/
│   └── evaluation/
├── models/        Trained models, predictions, metrics, and plots
├── dashboards/
│   ├── powerbi/
│   └── tableau/
├── docs/          Project documentation and setup guides
└── tests/         Automated tests
```

For detailed setup and technical information, see the documents in `docs/`, especially `KAIROS_documentation.pdf` and `Kairos_API_Contracts.md`.
