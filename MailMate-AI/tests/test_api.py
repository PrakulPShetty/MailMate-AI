"""API tests for the MailMate AI Flask backend.

Run from the repository root::

    pytest MailMate-AI/tests/test_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from backend.app import create_app  # noqa: E402
from backend.loader import ModelLoadError  # noqa: E402


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as test_client:
        yield test_client


@pytest.fixture(scope="module")
def categories(client):
    return client.get("/categories").get_json()["categories"]


# --------------------------------------------------------------------- /health
def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["version"] == "1.0.0"
    assert isinstance(body["uptime_seconds"], (int, float))


# ---------------------------------------------------------------- /categories
def test_categories(client):
    response = client.get("/categories")
    assert response.status_code == 200
    body = response.get_json()
    assert len(body["categories"]) == 7
    assert body["tones"] == ["professional", "formal", "polite"]
    assert body["max_suggestions"] == 3
    assert body["max_text_len"] == 2000


# ------------------------------------------------------------------- /predict
def test_predict_next_word(client, categories):
    response = client.post(
        "/predict", json={"text": "I am writing to follow up regarding my internship "}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["category"] in categories
    assert 1 <= len(body["suggestions"]) <= 3
    assert body["tone"]["label"] in ("professional", "formal", "polite")
    assert body["latency_ms"] >= 0
    for suggestion in body["suggestions"]:
        assert set(suggestion) == {"text", "score", "source"}
        assert suggestion["source"] == "mixed"


def test_predict_partial_word(client):
    response = client.post("/predict", json={"text": "Please find attached my res"})
    assert response.status_code == 200
    suggestions = response.get_json()["suggestions"]
    assert suggestions, "expected prefix completions for 'res'"
    assert all(item["text"].startswith("res") for item in suggestions)


def test_predict_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    assert len(response.get_json()["suggestions"]) == 3


def test_predict_with_explicit_category_and_tone(client):
    response = client.post(
        "/predict",
        json={"text": "Could you please ", "category": "meeting_request", "tone": "polite"},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["category"] == "meeting_request"
    assert body["tone"]["label"] == "polite"


def test_predict_cursor_pos(client):
    response = client.post(
        "/predict", json={"text": "I am writing to follow", "cursorPos": 5}
    )
    assert response.status_code == 200
    assert response.get_json()["category"]  # classification still returned


def test_predict_invalid_category(client):
    response = client.post("/predict", json={"text": "hello ", "category": "nope"})
    assert response.status_code == 400
    assert "category" in response.get_json()["message"].lower()


def test_predict_invalid_tone(client):
    response = client.post("/predict", json={"text": "hello ", "tone": "shouty"})
    assert response.status_code == 400
    assert "tone" in response.get_json()["message"].lower()


def test_predict_text_too_long(client):
    response = client.post("/predict", json={"text": "a" * 2001})
    assert response.status_code == 400
    assert response.get_json()["error"] == "text_too_long"


def test_predict_text_exactly_at_limit_is_allowed(client):
    response = client.post("/predict", json={"text": "a " * 1000})
    assert response.status_code == 200


def test_predict_invalid_text_type(client):
    response = client.post("/predict", json={"text": 123})
    assert response.status_code == 400


def test_predict_invalid_cursor_pos(client):
    response = client.post("/predict", json={"text": "hello ", "cursorPos": "5"})
    assert response.status_code == 400


def test_predict_missing_body(client):
    response = client.post("/predict", data="", content_type="application/json")
    assert response.status_code == 400


def test_predict_non_object_body(client):
    response = client.post("/predict", json=["not", "an", "object"])
    assert response.status_code == 400


# ------------------------------------------------------------------ /classify
def test_classify(client, categories):
    response = client.post(
        "/classify", json={"text": "Please find attached my resume for the position"}
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["category"] == "job_application"
    assert body["category"] in categories
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"]) == set(categories)


def test_classify_text_too_long(client):
    response = client.post("/classify", json={"text": "a" * 2001})
    assert response.status_code == 400


def test_classify_invalid_text_type(client):
    response = client.post("/classify", json={"text": None})
    assert response.status_code == 400


# --------------------------------------------------------------- errors / CORS
def test_unknown_route_returns_json_404(client):
    response = client.get("/nope")
    assert response.status_code == 404
    assert response.is_json
    assert response.get_json()["status"] == 404


def test_method_not_allowed_returns_json_405(client):
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.is_json


def test_cors_headers_present(client):
    response = client.get("/health")
    assert response.headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in response.headers["Access-Control-Allow-Methods"]


# ---------------------------------------------------------------- loader guard
def test_loader_is_cached_singleton():
    from backend import loader

    assert loader.get_predictor() is loader.get_predictor()
    assert loader.is_loaded() is True


def test_loader_rejects_missing_artifact(monkeypatch):
    from backend import loader

    monkeypatch.setattr(loader, "MODEL_PATH", Path("does/not/exist.joblib"))
    monkeypatch.setattr(loader, "_predictor", None)
    with pytest.raises(ModelLoadError):
        loader.get_predictor()
    assert loader.is_loaded() is False
