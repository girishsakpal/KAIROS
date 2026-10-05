"""
Kairos API — application factory.

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


def create_app(overrides=None):
    app = Flask(__name__)
    app.json.sort_keys = False
    app.config.from_object(Config)
    if overrides:
        app.config.update(overrides)

    app.extensions["kairos_repo"] = CsvRepository(app.config)
    register_error_handlers(app)
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    create_app().run(debug=True)
