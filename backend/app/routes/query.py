"""``POST /query`` - ask a question over the saved content."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_rag_service
from app.models import QueryRequest, QueryResponse
from app.services.rag import RagService

router = APIRouter(tags=["rag"])


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question and get an answer with cited sources",
    responses={
        422: {"description": "Blank question, or top_k outside 1-20."},
        429: {"description": "The AI provider rate limited the request."},
        502: {"description": "The AI provider failed."},
        503: {"description": "No OPENAI_API_KEY is configured."},
    },
)
def query_content(
    request: QueryRequest,
    service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    """Answer from saved content only.

    The `[n]` markers in `answer` line up with the `citation` field of each entry
    in `sources`, so every claim can be traced back to the text it came from.
    """
    return service.answer(request)
