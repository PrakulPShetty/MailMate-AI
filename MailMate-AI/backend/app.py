"""MailMate AI Flask REST API (CPU-only).

Serves the four endpoints consumed by ``frontend/lib/api.js``. Run with::

    python MailMate-AI/backend/app.py      # from the repository root
    python backend/app.py                  # from MailMate-AI/

The model is loaded eagerly at startup so the first HTTP request is already
warm (the predictor's own latency is ~1-3 ms).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from flask import Flask, Response, jsonify, request
from werkzeug.exceptions import HTTPException

# Windows consoles default to cp1252, which cannot encode the project path.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend import loader  # noqa: E402

__all__ = ["create_app", "app"]

MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # 1 MB is ample for 2000 characters.

STARTED_AT = time.monotonic()


def _json_error(status: int, error: str, message: str) -> tuple[Response, int]:
    response = jsonify({"error": error, "message": message, "status": status})
    response.status_code = status
    return response


def _json_body() -> tuple[dict | None, tuple[Response, int] | None]:
    """Parse a JSON object body, or return a ready-made 400 response."""
    if not request.data:
        return None, _json_error(400, "bad_request", "Request body is required.")
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return None, _json_error(400, "bad_request", "Request body must be a JSON object.")
    return payload, None


def _validate_text(payload: dict, predictor) -> tuple[str | None, tuple[Response, int] | None]:
    text = payload.get("text")
    if not isinstance(text, str):
        return None, _json_error(400, "bad_request", "'text' must be a string.")
    if len(text) > predictor.max_text_len:
        return None, _json_error(
            400,
            "text_too_long",
            f"'text' exceeds the {predictor.max_text_len}-character limit.",
        )
    return text, None


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
    app.config["JSON_SORT_KEYS"] = False

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Eager load: surfaces a bad artifact at startup, not mid-request.
    try:
        loader.get_predictor()
        app.logger.info("Model artifact loaded from %s", loader.MODEL_PATH)
    except loader.ModelLoadError as exc:
        app.logger.error("Model unavailable at startup: %s", exc)

    def predictor_or_error():
        try:
            return loader.get_predictor(), None
        except loader.ModelLoadError as exc:
            return None, _json_error(503, "model_unavailable", str(exc))

    # ------------------------------------------------------------------ routes
    @app.get("/health")
    def health():
        try:
            predictor = loader.get_predictor()
        except loader.ModelLoadError as exc:
            return _json_error(503, "model_unavailable", str(exc))
        return jsonify(
            {
                "status": "ok",
                "model_loaded": True,
                "version": predictor.version,
                "uptime_seconds": round(time.monotonic() - STARTED_AT, 3),
            }
        )

    @app.get("/categories")
    def categories():
        predictor, error = predictor_or_error()
        if error is not None:
            return error
        return jsonify(
            {
                "categories": predictor.categories,
                "tones": predictor.tones,
                "max_suggestions": predictor.max_suggestions,
                "max_text_len": predictor.max_text_len,
            }
        )

    @app.post("/predict")
    def predict():
        predictor, error = predictor_or_error()
        if error is not None:
            return error

        payload, error = _json_body()
        if error is not None:
            return error
        text, error = _validate_text(payload, predictor)
        if error is not None:
            return error

        cursor_pos = payload.get("cursorPos")
        if cursor_pos is not None and (
            isinstance(cursor_pos, bool) or not isinstance(cursor_pos, int)
        ):
            return _json_error(400, "bad_request", "'cursorPos' must be an integer or null.")

        category = payload.get("category")
        if category is not None and not isinstance(category, str):
            return _json_error(400, "bad_request", "'category' must be a string or null.")

        tone = payload.get("tone")
        if tone is not None and not isinstance(tone, str):
            return _json_error(400, "bad_request", "'tone' must be a string or null.")

        try:
            result = predictor.predict(
                text,
                cursor_pos=cursor_pos,
                category=category,
                tone=tone,
                top_k=predictor.max_suggestions,
            )
        except ValueError as exc:
            return _json_error(400, "invalid_parameter", str(exc))

        return jsonify(result)

    @app.post("/classify")
    def classify():
        predictor, error = predictor_or_error()
        if error is not None:
            return error

        payload, error = _json_body()
        if error is not None:
            return error
        text, error = _validate_text(payload, predictor)
        if error is not None:
            return error

        return jsonify(predictor.classify(text))

    # ------------------------------------------------------------ error / CORS
    @app.errorhandler(HTTPException)
    def handle_http_exception(exc: HTTPException):
        response = exc.get_response()
        response.data = json.dumps(
            {
                "error": (exc.name or "error").lower().replace(" ", "_"),
                "message": exc.description,
                "status": exc.code,
            }
        )
        response.content_type = "application/json"
        return response

    @app.errorhandler(Exception)
    def handle_unexpected_exception(exc: Exception):
        app.logger.exception("Unhandled error: %s", exc)
        return _json_error(500, "internal_server_error", "An unexpected error occurred.")

    @app.after_request
    def add_cors_headers(response: Response) -> Response:
        response.headers.setdefault("Access-Control-Allow-Origin", "*")
        response.headers.setdefault("Access-Control-Allow-Headers", "Content-Type")
        response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        response.headers.setdefault("Access-Control-Max-Age", "600")
        return response

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
