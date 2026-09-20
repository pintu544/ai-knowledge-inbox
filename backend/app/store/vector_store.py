"""In-memory item and vector storage.

Design notes / tradeoffs
------------------------
Single-user app, so storage is a plain dict plus one dense numpy matrix of chunk
embeddings, and search is an exhaustive cosine scan. That is O(n) per query, but
n here is "chunks one person saved", where a linear scan over a few thousand
1536-dim vectors is sub-millisecond and beats the complexity of an index.

Vectors are L2-normalised on insert, which turns cosine similarity into a single
matrix-vector dot product.

Everything lives in process memory and is intentionally lost on restart. See the
Tradeoffs section of the README for what changes when this needs to persist.
"""

from __future__ import annotations

import logging
import threading
from typing import Iterable

import numpy as np

from app.models import Chunk, Item, ScoredChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """Thread-safe in-memory store for items and their chunk embeddings."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._items: dict[str, Item] = {}
        self._chunks: list[Chunk] = []
        #: Row i of this matrix is the normalised embedding of ``self._chunks[i]``.
        self._matrix: np.ndarray | None = None
        self._dimension: int | None = None

    # --- writes ------------------------------------------------------------
    def add_item(self, item: Item, chunks: Iterable[Chunk], embeddings: Iterable[Iterable[float]]) -> Item:
        """Store an item together with its chunks and their embeddings.

        The three arguments must line up: ``chunks[i]`` is described by
        ``embeddings[i]``. Raises ``ValueError`` on a mismatch so a partial write
        can never silently corrupt retrieval.
        """
        chunk_list = list(chunks)
        embedding_list = [list(vector) for vector in embeddings]

        if len(chunk_list) != len(embedding_list):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunk_list)} chunks vs {len(embedding_list)} embeddings"
            )
        if not chunk_list:
            raise ValueError("cannot store an item with no chunks")

        dimensions = {len(vector) for vector in embedding_list}
        if len(dimensions) != 1:
            raise ValueError(f"embeddings have inconsistent dimensions: {sorted(dimensions)}")
        dimension = dimensions.pop()

        with self._lock:
            if self._dimension is not None and dimension != self._dimension:
                raise ValueError(
                    f"embedding dimension {dimension} does not match the stored dimension {self._dimension}; "
                    "clear the store after changing OPENAI_EMBEDDING_MODEL"
                )

            rows = _normalise_rows(np.asarray(embedding_list, dtype=np.float32))
            self._matrix = rows if self._matrix is None else np.vstack((self._matrix, rows))
            self._dimension = dimension

            stored = item.model_copy(update={"chunk_count": len(chunk_list)})
            self._items[stored.id] = stored
            self._chunks.extend(chunk_list)

        logger.info(
            "stored item",
            extra={
                "item_id": stored.id,
                "source_type": stored.source_type.value,
                "chunk_count": len(chunk_list),
                "total_chunks": len(self._chunks),
            },
        )
        return stored

    def clear(self) -> None:
        """Drop everything. Used by tests and available for a future reset endpoint."""
        with self._lock:
            self._items.clear()
            self._chunks.clear()
            self._matrix = None
            self._dimension = None

    # --- reads -------------------------------------------------------------
    def list_items(self) -> list[Item]:
        """All items, newest first (ties broken by id for a stable order)."""
        with self._lock:
            return sorted(self._items.values(), key=lambda item: (item.created_at, item.id), reverse=True)

    def get_item(self, item_id: str) -> Item | None:
        with self._lock:
            return self._items.get(item_id)

    @property
    def item_count(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def chunk_count(self) -> int:
        with self._lock:
            return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return self.chunk_count == 0

    def search(
        self,
        query_embedding: Iterable[float],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[ScoredChunk]:
        """Return up to ``top_k`` chunks most similar to ``query_embedding``.

        Results are ordered by descending cosine similarity. Chunks scoring
        below ``min_score`` are dropped, so a weak, off-topic match is never
        returned just to fill the top-k quota (this can return fewer than
        ``top_k`` results, or none). An empty store returns an empty list rather
        than raising, so callers can answer "nothing saved yet" gracefully.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        query = np.asarray(list(query_embedding), dtype=np.float32)

        with self._lock:
            if self._matrix is None or not self._chunks:
                return []
            if query.shape[0] != self._matrix.shape[1]:
                raise ValueError(
                    f"query embedding has dimension {query.shape[0]}, "
                    f"but stored vectors have dimension {self._matrix.shape[1]}"
                )

            # Both sides are unit vectors, so the dot product *is* cosine similarity.
            scores = self._matrix @ _normalise_vector(query)
            limit = min(top_k, scores.shape[0])
            # argpartition finds the top-k in O(n); only that slice gets sorted.
            candidates = np.argpartition(-scores, limit - 1)[:limit]
            ranked = candidates[np.argsort(-scores[candidates], kind="stable")]

            return [
                ScoredChunk(chunk=self._chunks[index], score=float(scores[index]))
                for index in ranked
                if float(scores[index]) >= min_score
            ]


def _normalise_vector(vector: np.ndarray) -> np.ndarray:
    """Scale to unit length; a zero vector is returned unchanged."""
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against divide-by-zero for degenerate all-zero embeddings.
    norms[norms == 0.0] = 1.0
    return matrix / norms
