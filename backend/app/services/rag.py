"""Retrieval-augmented answering.

Pipeline: embed the question -> cosine top-K over stored chunks -> build a
numbered context block -> ask the model to answer *only* from that context and
cite with [n] markers -> return the answer alongside the chunks it cited.

The numbered-context convention is what makes citations verifiable: the [n] in
the answer maps to ``sources[n-1]`` in the response, so the UI can show the exact
text the claim came from.
"""

from __future__ import annotations

import logging

from app.clients.openai_client import OpenAIClient
from app.config import Settings
from app.models import QueryRequest, QueryResponse, ScoredChunk, SourceSnippet
from app.store.vector_store import VectorStore

logger = logging.getLogger(__name__)

SNIPPET_MAX_CHARS = 400

EMPTY_STORE_ANSWER = (
    "Nothing has been saved yet, so there is no content to answer from. "
    "Add a note or a URL first."
)

NO_RELEVANT_CONTENT_ANSWER = (
    "None of your saved content is relevant enough to answer that. "
    "Try rephrasing, or save something on this topic first."
)

SYSTEM_PROMPT = """You answer questions strictly from the numbered context supplied by the user.

Rules:
- Use only facts present in the context. Never rely on outside knowledge.
- Cite the context you used with bracketed numbers such as [1] or [2, 3], placed
  immediately after the claim they support.
- If the context does not contain the answer, say so plainly and do not guess.
- Be concise and direct. No preamble, no restating the question."""


class RagService:
    """Answers questions over the ingested corpus."""

    def __init__(self, store: VectorStore, ai_client: OpenAIClient, settings: Settings) -> None:
        self._store = store
        self._ai = ai_client
        self._settings = settings

    def answer(self, request: QueryRequest) -> QueryResponse:
        """Retrieve context and produce a cited answer.

        An empty corpus returns a plain explanation instead of calling the model,
        which saves a pointless request and gives the UI something honest to show.

        Raises:
            OpenAIClientError: embedding or completion failed (translated to HTTP
                by the exception handlers).
        """
        if self._store.is_empty:
            logger.info("query on empty store", extra={"question_chars": len(request.question)})
            return QueryResponse(
                question=request.question,
                answer=EMPTY_STORE_ANSWER,
                sources=[],
                model=self._ai.chat_model,
                retrieved_chunk_count=0,
            )

        top_k = request.top_k or self._settings.retrieval_top_k
        question_embedding = self._ai.embed_text(request.question)
        retrieved = self._store.search(
            question_embedding,
            top_k=top_k,
            min_score=self._settings.retrieval_min_score,
        )

        logger.info(
            "retrieved context",
            extra={
                "top_k": top_k,
                "min_score": self._settings.retrieval_min_score,
                "retrieved": len(retrieved),
                "best_score": round(retrieved[0].score, 4) if retrieved else None,
            },
        )

        # Everything was below the relevance floor: answer honestly instead of
        # feeding the model an empty context and letting it improvise.
        if not retrieved:
            return QueryResponse(
                question=request.question,
                answer=NO_RELEVANT_CONTENT_ANSWER,
                sources=[],
                model=self._ai.chat_model,
                retrieved_chunk_count=0,
            )

        sources = self._build_sources(retrieved)
        answer = self._ai.complete_chat(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=_build_user_prompt(request.question, retrieved),
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            model=self._ai.chat_model,
            retrieved_chunk_count=len(retrieved),
        )

    def _build_sources(self, retrieved: list[ScoredChunk]) -> list[SourceSnippet]:
        """Attach item metadata to each retrieved chunk, numbered to match the prompt."""
        sources: list[SourceSnippet] = []

        for citation, scored in enumerate(retrieved, start=1):
            item = self._store.get_item(scored.chunk.item_id)
            if item is None:
                # Should not happen: chunks are only stored alongside their item.
                logger.warning("retrieved chunk with no parent item", extra={"chunk_id": scored.chunk.id})
                continue

            sources.append(
                SourceSnippet(
                    citation=citation,
                    item_id=item.id,
                    source_type=item.source_type,
                    title=item.title,
                    source_url=item.source_url,
                    chunk_index=scored.chunk.index,
                    score=round(scored.score, 4),
                    snippet=_truncate(scored.chunk.text, SNIPPET_MAX_CHARS),
                )
            )

        return sources


def _build_user_prompt(question: str, retrieved: list[ScoredChunk]) -> str:
    """Render the numbered context block followed by the question."""
    blocks = [f"[{citation}] {scored.chunk.text}" for citation, scored in enumerate(retrieved, start=1)]
    context = "\n\n".join(blocks) if blocks else "(no context available)"

    return f"Context:\n{context}\n\nQuestion: {question}"


def _truncate(text: str, limit: int) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + "..."
