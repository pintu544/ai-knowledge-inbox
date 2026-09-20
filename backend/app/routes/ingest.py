"""``POST /ingest`` - save a note or a URL."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.dependencies import get_ingestion_service
from app.models import IngestRequest, IngestResponse, ItemSummary
from app.services.ingestion import IngestionService

router = APIRouter(tags=["content"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save a note or URL and index it for retrieval",
    responses={
        422: {"description": "Blank text, or a malformed URL."},
        429: {"description": "The AI provider rate limited the embedding request."},
        502: {"description": "The URL could not be fetched, or the AI provider failed."},
        503: {"description": "No OPENAI_API_KEY is configured."},
    },
)
def ingest_content(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    """Store content and embed it so `/query` can retrieve it.

    For `source_type=url` the page is fetched server-side and reduced to its main
    article text before chunking.
    """
    item = service.ingest(request)
    return IngestResponse(item=ItemSummary.from_item(item), chunk_count=item.chunk_count)
