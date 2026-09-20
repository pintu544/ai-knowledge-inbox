"""Ingestion pipeline: raw input in, stored + embedded item out.

One place owns the whole sequence (resolve content -> chunk -> embed -> store) so
the route stays thin and the flow is readable end to end.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.clients.openai_client import OpenAIClient
from app.config import Settings
from app.errors import InvalidInputError, UpstreamFetchError
from app.models import Chunk, IngestRequest, Item, SourceType
from app.services.chunker import chunk_text
from app.services.url_extractor import UrlExtractionError, fetch_and_extract
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

NOTE_TITLE_MAX_CHARS = 70


@dataclass(frozen=True)
class ResolvedContent:
    """Content ready to chunk, whatever the original source was."""

    title: str
    text: str
    source_url: str | None


class IngestionService:
    """Turns an ingest request into a stored, searchable item."""

    def __init__(self, store: VectorStore, ai_client: OpenAIClient, settings: Settings) -> None:
        self._store = store
        self._ai = ai_client
        self._settings = settings

    def ingest(self, request: IngestRequest) -> Item:
        """Save a note or URL and index it for retrieval.

        Raises:
            InvalidInputError: the request cannot produce usable content.
            UpstreamFetchError: a URL could not be fetched or held no article text.
            OpenAIClientError: embedding failed (translated to HTTP by the handlers).
        """
        resolved = self._resolve_content(request)

        chunks_text = chunk_text(
            resolved.text,
            chunk_size=self._settings.chunk_size,
            overlap=self._settings.chunk_overlap,
        )
        if not chunks_text:
            raise InvalidInputError("There was no usable text to save after cleaning the input.")

        item = Item(
            source_type=request.source_type,
            title=resolved.title,
            source_url=resolved.source_url,
            content=resolved.text,
        )
        chunks = [Chunk(item_id=item.id, index=index, text=text) for index, text in enumerate(chunks_text)]

        logger.info(
            "embedding item",
            extra={
                "item_id": item.id,
                "source_type": item.source_type.value,
                "chunk_count": len(chunks),
                "chars": len(resolved.text),
            },
        )
        embeddings = self._ai.embed_texts([chunk.text for chunk in chunks])

        try:
            return self._store.add_item(item, chunks, embeddings)
        except ValueError as error:
            # A mismatch here is a bug on our side, not bad user input.
            logger.error("failed to store item", extra={"item_id": item.id, "detail": str(error)})
            raise

    # --- content resolution ------------------------------------------------
    def _resolve_content(self, request: IngestRequest) -> ResolvedContent:
        if request.source_type is SourceType.URL:
            return self._resolve_url(request)
        return self._resolve_note(request)

    def _resolve_note(self, request: IngestRequest) -> ResolvedContent:
        return ResolvedContent(
            title=request.title or _derive_note_title(request.content),
            text=request.content,
            source_url=None,
        )

    def _resolve_url(self, request: IngestRequest) -> ResolvedContent:
        try:
            url = request.validated_url()
        except ValueError as error:
            raise InvalidInputError(str(error), {"field": "content"}) from error

        try:
            page = fetch_and_extract(
                url,
                timeout_seconds=self._settings.url_fetch_timeout_seconds,
                max_chars=self._settings.max_content_chars,
            )
        except UrlExtractionError as error:
            logger.warning("url ingestion failed", extra={"url": url, "detail": str(error)})
            raise UpstreamFetchError(str(error), {"url": url}) from error

        return ResolvedContent(
            title=request.title or page.title or _derive_url_title(page.url),
            text=page.text,
            source_url=page.url,
        )


def _derive_note_title(content: str) -> str:
    """Use the note's opening words as its title."""
    first_line = next((line.strip() for line in content.splitlines() if line.strip()), "")
    collapsed = " ".join(first_line.split()) or "Untitled note"

    if len(collapsed) <= NOTE_TITLE_MAX_CHARS:
        return collapsed

    truncated = collapsed[:NOTE_TITLE_MAX_CHARS].rsplit(" ", 1)[0]
    return f"{truncated or collapsed[:NOTE_TITLE_MAX_CHARS]}..."


def _derive_url_title(url: str) -> str:
    """Fall back to the hostname when a page has no usable title."""
    from urllib.parse import urlparse

    host = urlparse(url).netloc
    return host or url
