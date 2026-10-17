"""
Kairos API — configuration.

Paths default to the repo layout. Override any value by passing a dict to
create_app(...) (the tests do this to point at tiny fixture files).

WHY THESE DEFAULTS
------------------
PREDICTIONS_FILE / SERVING_REGISTRY_FILE point at the XGBoost outputs, not the
champion's. The champion (currently the LSTM) has no per-customer SHAP drivers,
and the API contract promises `top_features` on every prediction. Same split the
Power BI dashboard uses: the LSTM scores best, XGBoost explains.
When the Postgres load exists, these become queries instead of file reads.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:                                   # let .env set KAIROS_REPOSITORY / DATABASE_URL
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


class Config:
    # "csv"      -> read data/ and models/ files (no database needed; the default)
    # "postgres" -> read Supabase/Postgres via DATABASE_URL (see docs/SUPABASE_SETUP.md)
    REPOSITORY = os.getenv("KAIROS_REPOSITORY", "csv").strip().lower()

    SYNTHETIC_DIR = ROOT / "data" / "synthetic"
    PROCESSED_DIR = ROOT / "data" / "processed"
    MODELS_DIR = ROOT / "models"

    PREDICTIONS_FILE = "xgboost_predictions_test.csv"
    SERVING_REGISTRY_FILE = "xgboost_registry_row.json"
    CHAMPION_REGISTRY_FILE = "model_registry_row.json"

    # Fallback "as-of" date for customers with no snapshot_date (must match the
    # generator's TODAY / build_features.REFERENCE_DATE).
    REFERENCE_DATE = "2026-09-16"

    # ASSUMPTION, not derived from data: a model older than this is flagged stale.
    STALE_AFTER_DAYS = 30

    # Injectable clock for tests; None means "use the real current time (UTC)".
    NOW = None
