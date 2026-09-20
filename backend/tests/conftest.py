"""Shared test fixtures.

The real repo-root ``.env`` holds a live API key, so these overrides are applied
at import time (before ``app`` is imported anywhere) to guarantee the suite can
never authenticate against OpenAI.
"""

from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = "test-key-not-real"
os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
os.environ["OPENAI_EMBEDDING_MODEL"] = "text-embedding-3-small"
os.environ["LOG_JSON"] = "false"
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.dependencies import get_openai_client, reset_dependency_caches  # noqa: E402
from app.main import create_app  # noqa: E402
from tests.fakes import FakeOpenAIClient  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_singletons():
    """Every test starts with fresh settings, client, and (empty) store."""
    reset_dependency_caches()
    yield
    reset_dependency_caches()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def fake_openai() -> FakeOpenAIClient:
    return FakeOpenAIClient()


@pytest.fixture
def client(fake_openai: FakeOpenAIClient) -> TestClient:
    """API client with the AI provider swapped for a deterministic fake."""
    app = create_app()
    app.dependency_overrides[get_openai_client] = lambda: fake_openai
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
