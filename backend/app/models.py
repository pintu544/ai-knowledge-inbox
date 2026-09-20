"""Pydantic schemas: the API contract and the internal domain objects.

Request models validate aggressively at the edge so services can assume clean
input. Response models are explicit (no leaking internal fields) and drive the
OpenAPI docs at ``/docs``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

PREVIEW_CHARS = 240


class SourceType(str, Enum):
    """Where a saved item came from."""

    NOTE = "note"
    URL = "url"


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    """Save a plain-text note or a URL to fetch server-side."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"source_type": "note", "content": "Postgres advisory locks are per-session, not per-transaction."},
                {"source_type": "url", "content": "https://fastapi.tiangolo.com/tutorial/first-steps/"},
            ]
        }
    )

    source_type: SourceType = Field(description="`note` for raw text, `url` to fetch and extract a web page.")
    content: str = Field(
        min_length=1,
        max_length=100_000,
        description="The note text, or the URL to fetch when `source_type` is `url`.",
    )
    title: str | None = Field(
        default=None,
        max_length=300,
        description="Optional label. For URLs, the extracted page title is used when omitted.",
    )

    @field_validator("content", "title", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("content")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("content cannot be empty or whitespace only")
        return value

    @field_validator("title")
    @classmethod
    def _blank_title_is_none(cls, value: str | None) -> str | None:
        return value or None

    def validated_url(self) -> str:
        """Return ``content`` as an http(s) URL, or raise ``ValueError``."""
        parsed = urlparse(self.content)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("URL must start with http:// or https://")
        if not parsed.netloc:
            raise ValueError("URL is missing a hostname")
        return self.content


class QueryRequest(BaseModel):
    """Ask a question over everything that has been ingested."""

    model_config = ConfigDict(
        json_schema_extra={"examples": [{"question": "What did I save about advisory locks?"}]}
    )

    question: str = Field(min_length=1, max_length=2_000, description="Natural-language question.")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="How many chunks to retrieve. Defaults to the server's RETRIEVAL_TOP_K.",
    )

    @field_validator("question", mode="before")
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("question")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if not value:
            raise ValueError("question cannot be empty or whitespace only")
        return value


# --------------------------------------------------------------------------- #
# Domain objects
# --------------------------------------------------------------------------- #
def _new_id() -> str:
    return uuid4().hex


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


class Item(BaseModel):
    """A saved piece of content plus its metadata."""

    id: str = Field(default_factory=_new_id)
    source_type: SourceType
    title: str
    source_url: str | None = None
    content: str
    created_at: datetime = Field(default_factory=_now)
    chunk_count: int = 0

    @property
    def preview(self) -> str:
        collapsed = " ".join(self.content.split())
        if len(collapsed) <= PREVIEW_CHARS:
            return collapsed
        return collapsed[:PREVIEW_CHARS].rstrip() + "..."


class Chunk(BaseModel):
    """A retrievable slice of an item's content."""

    id: str = Field(default_factory=_new_id)
    item_id: str
    index: int = Field(ge=0, description="Position of this chunk within its item.")
    text: str


class ScoredChunk(BaseModel):
    """A chunk plus its similarity to a question."""

    chunk: Chunk
    score: float


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class ItemSummary(BaseModel):
    """Item as returned by the API: metadata plus a short preview, not full text."""

    id: str
    source_type: SourceType
    title: str
    source_url: str | None
    created_at: datetime
    chunk_count: int
    char_count: int
    preview: str

    @classmethod
    def from_item(cls, item: Item) -> "ItemSummary":
        return cls(
            id=item.id,
            source_type=item.source_type,
            title=item.title,
            source_url=item.source_url,
            created_at=item.created_at,
            chunk_count=item.chunk_count,
            char_count=len(item.content),
            preview=item.preview,
        )


class IngestResponse(BaseModel):
    """Result of a successful ingestion."""

    item: ItemSummary
    chunk_count: int


class ItemListResponse(BaseModel):
    """All saved items, newest first."""

    items: list[ItemSummary]
    count: int


class SourceSnippet(BaseModel):
    """A cited chunk backing an answer."""

    citation: int = Field(ge=1, description="Matches the [n] markers used in the answer.")
    item_id: str
    source_type: SourceType
    title: str
    source_url: str | None
    chunk_index: int
    score: float
    snippet: str


class QueryResponse(BaseModel):
    """An answer plus the sources it was built from."""

    question: str
    answer: str
    sources: list[SourceSnippet]
    model: str = Field(description="Chat model that produced the answer.")
    retrieved_chunk_count: int


class ModelStatus(BaseModel):
    """Configured vs. actually-used chat model."""

    configured: str
    in_use: str
    validated: bool = Field(description="True when the model was confirmed available for this API key.")
    detail: str | None = None


class HealthResponse(BaseModel):
    status: str
    chat_model: ModelStatus
    embedding_model: str
