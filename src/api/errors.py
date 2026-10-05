"""Shared error format from docs/Kairos_API_Contracts.md:
{"error": {"code": "...", "message": "...", "details": {}}}"""
from flask import jsonify
from werkzeug.exceptions import HTTPException

_CODES = {
    400: "BAD_REQUEST", 404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED",
    415: "UNSUPPORTED_MEDIA_TYPE", 422: "VALIDATION_ERROR",
}


class ApiError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.details = details or {}


def _body(code, message, details=None):
    return {"error": {"code": code, "message": message, "details": details or {}}}


def register_error_handlers(app):
    @app.errorhandler(ApiError)
    def _api_error(e):
        return jsonify(_body(e.code, e.message, e.details)), e.status

    @app.errorhandler(HTTPException)
    def _http_error(e):
        return jsonify(_body(_CODES.get(e.code, e.name.upper().replace(" ", "_")),
                              e.description)), e.code

    @app.errorhandler(Exception)
    def _unhandled(e):
        app.logger.exception("Unhandled error")
        return jsonify(_body("INTERNAL_ERROR", "Unexpected server error.")), 500
