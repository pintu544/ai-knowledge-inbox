"""FastAPI application factory and startup wiring."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.dependencies import get_openai_client
from app.errors import register_exception_handlers
from app.logging_config import configure_logging
from app.routes import health, ingest, items, query

logger = logging.getLogger(__name__)

API_DESCRIPTION = """
Save notes and URLs, then ask questions over them. Answers are generated with a
retrieval-augmented pipeline and always cite the chunks they came from.

Storage is in-memory and **ephemeral**: restarting the server clears everything.
"""


def _validate_chat_model(app: FastAPI) -> None:
    """Resolve the configured chat model once, at boot.

    Never raises: an unreachable model degrades to the default and is reported
    via logs and ``GET /health`` instead of failing every later request.
    """
    # Honour dependency overrides so tests (and any future harness) validate
    # against their injected client rather than reaching for the network.
    provider = app.dependency_overrides.get(get_openai_client, get_openai_client)
    client = provider()
    resolution = client.resolve_chat_model()
    app.state.model_resolution = resolution

    context = {
        "configured_model": resolution.requested,
        "model_in_use": resolution.resolved,
        "validated": resolution.available,
    }

    if resolution.available:
        logger.info("chat model validated against OpenAI", extra=context)
    elif resolution.used_fallback:
        logger.warning(
            "configured chat model %r is not available for this API key; falling back to %r",
            resolution.requested,
            resolution.resolved,
            extra={**context, "detail": resolution.detail},
        )
    else:
        logger.warning(
            "chat model %r could not be validated; continuing with it anyway",
            resolution.requested,
            extra={**context, "detail": resolution.detail},
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()

    if not settings.has_openai_credentials:
        logger.warning(
            "OPENAI_API_KEY is not set: /ingest and /query will return 503 until it is configured",
            extra={"embedding_model": settings.openai_embedding_model},
        )

    _validate_chat_model(app)
    logger.info(
        "ai knowledge inbox started",
        extra={"version": __version__, "storage": "in-memory (ephemeral)"},
    )
    yield
    logger.info("ai knowledge inbox shutting down; in-memory content discarded")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts settings so tests can inject their own."""
    settings = settings or get_settings()
    configure_logging(level=settings.log_level, as_json=settings.log_json)

    app = FastAPI(
        title="AI Knowledge Inbox",
        description=API_DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ingest.router)
    app.include_router(items.router)
    app.include_router(query.router)

    return app


app = create_app()
