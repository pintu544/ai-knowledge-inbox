"""Task 1: settings load from the environment with safe defaults."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL, Settings


def test_reads_values_from_the_environment(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("RETRIEVAL_TOP_K", "7")

    settings = Settings()

    assert settings.openai_model == "gpt-4o"
    assert settings.retrieval_top_k == 7


def test_blank_model_falls_back_to_the_verified_default(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "   ")
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "")

    settings = Settings()

    assert settings.openai_model == DEFAULT_CHAT_MODEL
    assert settings.openai_embedding_model == DEFAULT_EMBEDDING_MODEL


def test_unverified_model_names_are_accepted_without_code_changes(monkeypatch):
    """Swapping models must be a one-line .env change, even for exotic names."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")

    assert Settings().openai_model == "gpt-5.6-luna"


def test_cors_origins_parse_into_a_list(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:5173, http://localhost:4173 ,")

    assert Settings().cors_origin_list == ["http://localhost:5173", "http://localhost:4173"]


def test_missing_key_is_reported_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")

    assert Settings().has_openai_credentials is False


@pytest.mark.parametrize(
    ("variable", "value"),
    [("CHUNK_SIZE", "0"), ("CHUNK_OVERLAP", "-1"), ("RETRIEVAL_TOP_K", "0")],
)
def test_nonsensical_tunables_are_rejected(monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError):
        Settings()


def test_overlap_must_be_smaller_than_chunk_size(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "500")

    with pytest.raises(ValidationError, match="smaller than CHUNK_SIZE"):
        Settings()
