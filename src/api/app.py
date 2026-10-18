"""
Kairos API — application factory.

Backend: KAIROS_REPOSITORY=csv (default) or postgres (reads DATABASE_URL), set in .env or the shell.

Run from the repo root (either works):
    python -m src.api.app
    flask --app src.api.app run --debug          # create_app is auto-detected
Then:  GET http://127.0.0.1:5000/api/v1/health
"""
from pathlib import Path

from flask import Flask, send_from_directory

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

    # Demo UI (Sem VII checkpoint): same-origin page that shows the dashboard and calls the API.
    ui_dir = Path(__file__).resolve().parent.parent / "ui"

    @app.get("/")
    def ui_index():
        return send_from_directory(ui_dir, "index.html")

    @app.get("/ui/<path:filename>")
    def ui_files(filename):
        return send_from_directory(ui_dir, filename)

    return app


if __name__ == "__main__":
    create_app().run(debug=True)
