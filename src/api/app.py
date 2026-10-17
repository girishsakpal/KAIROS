"""
Kairos API — application factory.

Backend: KAIROS_REPOSITORY=csv (default) or postgres (reads DATABASE_URL), set in .env or the shell.

Run from the repo root (either works):
    python -m src.api.app
    flask --app src.api.app run --debug          # create_app is auto-detected
Then:  GET http://127.0.0.1:5000/api/v1/health
"""
from flask import Flask

from .config import Config
from .errors import register_error_handlers
from .repository import CsvRepository
from .routes import bp


def create_app(overrides=None, repository=None):
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    if repository is None:
        backend = app.config["REPOSITORY"]
        if backend == "postgres":
            from .repository_pg import PostgresRepository
            repository = PostgresRepository.from_env(app.config["REFERENCE_DATE"])
        elif backend == "csv":
            repository = CsvRepository(app.config)
        else:
            raise ValueError(f"KAIROS_REPOSITORY must be 'csv' or 'postgres', got {backend!r}")
    app.extensions["kairos_repo"] = repository
    register_error_handlers(app)
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
