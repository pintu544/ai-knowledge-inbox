"""Task 1: the app boots, reports health, and exposes its resolved models."""

from __future__ import annotations


def test_health_returns_ok_and_resolved_models(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["embedding_model"] == "text-embedding-3-small"
    assert body["chat_model"]["configured"] == "gpt-4o-mini"
    assert body["chat_model"]["in_use"] == "gpt-4o-mini"
    assert body["chat_model"]["validated"] is True


def test_openapi_schema_is_served(client):
    response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]


def test_unknown_route_uses_the_shared_error_envelope(client):
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
