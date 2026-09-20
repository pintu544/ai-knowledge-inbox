"""Task 1: startup model validation, including the fallback path.

This is the safety net for the unverified model names in the brief
(`gpt-5.6-luna` and friends): if the configured model is not available to the
API key, the app says so loudly and keeps working on the default.
"""

from __future__ import annotations

from app.clients.openai_client import OpenAIClient
from app.config import DEFAULT_CHAT_MODEL, Settings
from tests.fakes import StubSdk, auth_error, model_not_found_error


def _client(sdk: StubSdk, **env) -> OpenAIClient:
    return OpenAIClient(Settings(**env), sdk=sdk)


def test_available_model_is_used_as_is(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    sdk = StubSdk()

    resolution = _client(sdk).resolve_chat_model()

    assert sdk.models.retrieved == ["gpt-4o"]
    assert resolution.available is True
    assert resolution.resolved == "gpt-4o"
    assert resolution.used_fallback is False


def test_unavailable_model_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    client = _client(StubSdk(retrieve_error=model_not_found_error()))

    resolution = client.resolve_chat_model()

    assert resolution.available is False
    assert resolution.requested == "gpt-5.6-luna"
    assert resolution.resolved == DEFAULT_CHAT_MODEL
    assert resolution.used_fallback is True
    assert "model name" in (resolution.detail or "")
    # Subsequent calls use the fallback, not the broken configured value.
    assert client.chat_model == DEFAULT_CHAT_MODEL


def test_default_model_is_kept_even_when_validation_fails(monkeypatch):
    """Nothing to fall back to, so keep going and report the problem."""
    monkeypatch.setenv("OPENAI_MODEL", DEFAULT_CHAT_MODEL)
    client = _client(StubSdk(retrieve_error=auth_error()))

    resolution = client.resolve_chat_model()

    assert resolution.available is False
    assert resolution.resolved == DEFAULT_CHAT_MODEL
    assert resolution.used_fallback is False
    assert "API key" in (resolution.detail or "")


def test_validation_is_skipped_without_credentials(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    sdk = StubSdk()

    resolution = _client(sdk).resolve_chat_model()

    assert sdk.models.retrieved == []
    assert resolution.available is False
    assert resolution.resolved == "gpt-5.6-luna"
    assert "OPENAI_API_KEY" in (resolution.detail or "")


def test_startup_records_resolution_on_app_state(client):
    """The fake client reports availability, and /health mirrors app.state."""
    assert client.app.state.model_resolution.available is True
    assert client.get("/health").json()["chat_model"]["validated"] is True
