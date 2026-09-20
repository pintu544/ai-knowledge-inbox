"""Dependency wiring.

Single-user app, so the store and clients are process-wide singletons. They are
exposed as FastAPI dependencies (rather than imported directly by routes) so
tests can override them and a future multi-tenant version has one place to change.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from app.clients.openai_client import OpenAIClient
from app.config import Settings, get_settings
from app.services.ingestion import IngestionService
from app.services.rag import RagService
from app.store.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_openai_client() -> OpenAIClient:
    return OpenAIClient(get_settings())


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """The single in-memory store for this process."""
    return VectorStore()


def get_ingestion_service(
    store: VectorStore = Depends(get_vector_store),
    ai_client: OpenAIClient = Depends(get_openai_client),
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    """Built per request: cheap, and it picks up dependency overrides in tests."""
    return IngestionService(store=store, ai_client=ai_client, settings=settings)


def get_rag_service(
    store: VectorStore = Depends(get_vector_store),
    ai_client: OpenAIClient = Depends(get_openai_client),
    settings: Settings = Depends(get_settings),
) -> RagService:
    return RagService(store=store, ai_client=ai_client, settings=settings)


def reset_dependency_caches() -> None:
    """Drop cached singletons. Used by tests to get a clean process state."""
    get_settings.cache_clear()
    get_openai_client.cache_clear()
    get_vector_store.cache_clear()


__all__ = [
    "Settings",
    "VectorStore",
    "get_settings",
    "get_openai_client",
    "get_vector_store",
    "get_ingestion_service",
    "get_rag_service",
    "reset_dependency_caches",
]
