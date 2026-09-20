"""Task 2: the in-memory store keeps items and ranks chunks correctly.

Embeddings here are hand-written so the expected ranking is unambiguous.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from app.models import Chunk, Item, SourceType
from app.store.vector_store import VectorStore


def _item(title: str = "note", source_type: SourceType = SourceType.NOTE, **overrides) -> Item:
    return Item(source_type=source_type, title=title, content=overrides.pop("content", "body text"), **overrides)


def _chunks(item_id: str, *texts: str) -> list[Chunk]:
    return [Chunk(item_id=item_id, index=index, text=text) for index, text in enumerate(texts)]


# --- writes ---------------------------------------------------------------- #
def test_add_item_records_chunk_count_and_is_retrievable():
    store = VectorStore()
    item = _item("advisory locks")

    stored = store.add_item(item, _chunks(item.id, "a", "b"), [[1.0, 0.0], [0.0, 1.0]])

    assert stored.chunk_count == 2
    assert store.item_count == 1
    assert store.chunk_count == 2
    assert store.get_item(item.id) is not None
    assert store.get_item(item.id).chunk_count == 2


def test_add_item_rejects_mismatched_chunks_and_embeddings():
    store = VectorStore()
    item = _item()

    with pytest.raises(ValueError, match="count mismatch"):
        store.add_item(item, _chunks(item.id, "a", "b"), [[1.0, 0.0]])

    assert store.item_count == 0
    assert store.chunk_count == 0


def test_add_item_rejects_an_empty_chunk_list():
    store = VectorStore()
    item = _item()

    with pytest.raises(ValueError, match="no chunks"):
        store.add_item(item, [], [])


def test_add_item_rejects_ragged_embeddings():
    store = VectorStore()
    item = _item()

    with pytest.raises(ValueError, match="inconsistent dimensions"):
        store.add_item(item, _chunks(item.id, "a", "b"), [[1.0, 0.0], [0.0, 1.0, 0.0]])


def test_add_item_rejects_a_changed_embedding_dimension():
    """Guards against mixing vectors from two different embedding models."""
    store = VectorStore()
    first = _item()
    store.add_item(first, _chunks(first.id, "a"), [[1.0, 0.0]])

    second = _item()
    with pytest.raises(ValueError, match="does not match the stored dimension"):
        store.add_item(second, _chunks(second.id, "b"), [[1.0, 0.0, 0.0]])


def test_clear_empties_the_store():
    store = VectorStore()
    item = _item()
    store.add_item(item, _chunks(item.id, "a"), [[1.0, 0.0]])

    store.clear()

    assert store.is_empty is True
    assert store.item_count == 0
    assert store.list_items() == []
    # A different dimension is accepted again once the store is empty.
    other = _item()
    store.add_item(other, _chunks(other.id, "b"), [[1.0, 0.0, 0.0]])
    assert store.chunk_count == 1


# --- listing --------------------------------------------------------------- #
def test_list_items_returns_newest_first():
    store = VectorStore()
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for offset, title in enumerate(["oldest", "middle", "newest"]):
        item = _item(title, created_at=base + timedelta(minutes=offset))
        store.add_item(item, _chunks(item.id, "text"), [[1.0, 0.0]])

    assert [item.title for item in store.list_items()] == ["newest", "middle", "oldest"]


def test_list_items_on_an_empty_store():
    assert VectorStore().list_items() == []


# --- search ---------------------------------------------------------------- #
def test_search_ranks_by_cosine_similarity():
    store = VectorStore()
    item = _item()
    # "east" is identical to the query, "diagonal" partially aligned, "north" orthogonal.
    store.add_item(
        item,
        _chunks(item.id, "north", "diagonal", "east"),
        [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
    )

    results = store.search([1.0, 0.0], top_k=3)

    assert [scored.chunk.text for scored in results] == ["east", "diagonal", "north"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.7071, abs=1e-4)
    assert results[2].score == pytest.approx(0.0)


def test_search_is_scale_invariant():
    """Magnitude must not beat direction: vectors are normalised on insert."""
    store = VectorStore()
    item = _item()
    store.add_item(
        item,
        _chunks(item.id, "loud but off-target", "quiet but on-target"),
        [[5.0, 5.0], [0.01, 0.0]],
    )

    results = store.search([1.0, 0.0], top_k=1)

    assert results[0].chunk.text == "quiet but on-target"


def test_search_honours_top_k():
    store = VectorStore()
    item = _item()
    store.add_item(item, _chunks(item.id, "a", "b", "c"), [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    assert len(store.search([1.0, 0.0], top_k=2)) == 2


def test_search_clamps_top_k_to_the_number_of_chunks():
    store = VectorStore()
    item = _item()
    store.add_item(item, _chunks(item.id, "only"), [[1.0, 0.0]])

    assert len(store.search([1.0, 0.0], top_k=10)) == 1


def test_search_spans_multiple_items():
    store = VectorStore()
    first = _item("first")
    second = _item("second")
    store.add_item(first, _chunks(first.id, "orthogonal"), [[0.0, 1.0]])
    store.add_item(second, _chunks(second.id, "aligned"), [[1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert results[0].chunk.item_id == second.id
    assert results[1].chunk.item_id == first.id


def test_search_on_an_empty_store_returns_nothing():
    assert VectorStore().search([1.0, 0.0], top_k=3) == []


def test_search_rejects_a_wrong_sized_query():
    store = VectorStore()
    item = _item()
    store.add_item(item, _chunks(item.id, "a"), [[1.0, 0.0]])

    with pytest.raises(ValueError, match="dimension"):
        store.search([1.0, 0.0, 0.0], top_k=1)


def test_search_rejects_a_non_positive_top_k():
    with pytest.raises(ValueError, match="top_k"):
        VectorStore().search([1.0, 0.0], top_k=0)


def test_zero_vectors_do_not_break_search():
    store = VectorStore()
    item = _item()
    store.add_item(item, _chunks(item.id, "empty-ish", "real"), [[0.0, 0.0], [1.0, 0.0]])

    results = store.search([1.0, 0.0], top_k=2)

    assert results[0].chunk.text == "real"
    assert results[1].score == pytest.approx(0.0)
