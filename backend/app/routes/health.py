"""Liveness / configuration introspection endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from app.clients.openai_client import OpenAIClient
from app.dependencies import get_openai_client
from app.models import HealthResponse, ModelStatus

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Service health and resolved configuration")
def read_health(
    request: Request,
    client: OpenAIClient = Depends(get_openai_client),
) -> HealthResponse:
    """Report liveness plus which models the service actually resolved at boot.

    Handy for answering "is my configured model real?" without reading logs.
    """
    resolution = getattr(request.app.state, "model_resolution", None)

    return HealthResponse(
        status="ok",
        chat_model=ModelStatus(
            configured=resolution.requested if resolution else client.chat_model,
            in_use=resolution.resolved if resolution else client.chat_model,
            validated=bool(resolution.available) if resolution else False,
            detail=resolution.detail if resolution else "Model validation has not run.",
        ),
        embedding_model=client.embedding_model,
    )
