"""``GET /items`` - list everything saved."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.dependencies import get_vector_store
from app.models import ItemListResponse, ItemSummary
from app.store.vector_store import VectorStore

router = APIRouter(tags=["content"])


@router.get(
    "/items",
    response_model=ItemListResponse,
    summary="List saved items, newest first",
)
def list_items(store: VectorStore = Depends(get_vector_store)) -> ItemListResponse:
    """Return item metadata and a short preview.

    Full text is deliberately omitted: the list view does not need it, and notes
    can be long. Snippets of the actual content come back from `/query` as
    citations.
    """
    items = [ItemSummary.from_item(item) for item in store.list_items()]
    return ItemListResponse(items=items, count=len(items))
