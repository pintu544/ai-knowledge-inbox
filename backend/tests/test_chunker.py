"""Task 3: chunking is intentional, lossless, and always makes progress."""

from __future__ import annotations

import pytest

from app.services.chunker import chunk_text, estimate_chunk_count, normalise_text


# --- degenerate input ------------------------------------------------------ #
@pytest.mark.parametrize("text", ["", "   ", "\n\t  \n"])
def test_blank_text_produces_no_chunks(text):
    assert chunk_text(text) == []


def test_short_text_becomes_a_single_chunk():
    assert chunk_text("Advisory locks are session scoped.") == ["Advisory locks are session scoped."]


def test_text_exactly_at_the_window_stays_one_chunk():
    text = "a" * 900

    assert chunk_text(text, chunk_size=900, overlap=150) == [text]


def test_one_character_over_the_window_splits():
    chunks = chunk_text("a" * 901, chunk_size=900, overlap=150)

    assert len(chunks) == 2


# --- normalisation --------------------------------------------------------- #
def test_whitespace_is_collapsed():
    assert normalise_text("  many\n\n spaces\there ") == "many spaces here"


def test_chunks_are_trimmed():
    chunks = chunk_text("  padded note  ")

    assert chunks == ["padded note"]


# --- windowing ------------------------------------------------------------- #
def test_long_text_is_split_into_overlapping_chunks():
    sentences = [f"Sentence number {index} explains an idea clearly. " for index in range(120)]
    chunks = chunk_text("".join(sentences), chunk_size=300, overlap=60)

    assert len(chunks) > 1
    assert all(len(chunk) <= 300 for chunk in chunks)


def test_consecutive_chunks_share_content():
    text = " ".join(f"word{index}" for index in range(400))
    chunks = chunk_text(text, chunk_size=300, overlap=90)

    assert len(chunks) >= 3
    for previous, current in zip(chunks, chunks[1:]):
        tail_words = set(previous.split()[-5:])
        assert tail_words & set(current.split()), "expected neighbouring chunks to overlap"


def test_no_content_is_lost():
    text = " ".join(f"token{index}" for index in range(500))
    chunks = chunk_text(text, chunk_size=250, overlap=50)

    seen = " ".join(chunks).split()
    for token in text.split():
        assert token in seen


def test_zero_overlap_is_allowed_and_does_not_duplicate():
    text = " ".join(f"w{index}" for index in range(200))
    chunks = chunk_text(text, chunk_size=120, overlap=0)

    rejoined = " ".join(chunks).split()
    assert len(rejoined) == len(set(rejoined)) == 200


# --- boundary snapping ----------------------------------------------------- #
def test_chunks_prefer_sentence_boundaries():
    text = ("First sentence is here. " * 20) + ("Second batch continues. " * 20)
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    # Every chunk but the last should end on sentence punctuation.
    assert all(chunk.endswith(".") for chunk in chunks[:-1])


def test_chunks_do_not_split_words_when_avoidable():
    text = " ".join("supercalifragilistic" for _ in range(60))
    chunks = chunk_text(text, chunk_size=200, overlap=40)

    for chunk in chunks:
        for word in chunk.split():
            assert word == "supercalifragilistic"


def test_an_unbroken_token_is_cut_hard_rather_than_looping_forever():
    chunks = chunk_text("x" * 1000, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert "".join(chunk for chunk in chunks).count("x") >= 1000


# --- guard rails ----------------------------------------------------------- #
@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size"),
        (-1, 0, "chunk_size"),
        (100, -1, "overlap cannot be negative"),
        (100, 100, "smaller than chunk_size"),
        (100, 150, "smaller than chunk_size"),
    ],
)
def test_invalid_parameters_are_rejected(chunk_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        chunk_text("some text that is long enough to matter", chunk_size=chunk_size, overlap=overlap)


def test_chunk_indices_are_contiguous_when_used_by_callers():
    """The chunker returns plain strings; ordering is positional and stable."""
    text = " ".join(f"w{index}" for index in range(300))

    assert chunk_text(text, chunk_size=200, overlap=40) == chunk_text(text, chunk_size=200, overlap=40)


# --- estimate -------------------------------------------------------------- #
def test_estimate_is_zero_for_blank_text():
    assert estimate_chunk_count("  ") == 0


def test_estimate_is_in_the_right_ballpark():
    text = "a " * 2000
    estimate = estimate_chunk_count(text, chunk_size=400, overlap=80)
    actual = len(chunk_text(text, chunk_size=400, overlap=80))

    assert abs(estimate - actual) <= 2
