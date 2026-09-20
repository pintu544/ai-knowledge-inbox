"""Task 7: retrieval picks the right chunks and answers cite their sources."""

from __future__ import annotations

import pytest

from app.services.rag import EMPTY_STORE_ANSWER
from tests.fakes import connection_error, rate_limit_error

LOCKS_NOTE = "Postgres advisory locks are session scoped and survive a transaction rollback."
DEPLOY_NOTE = "The deployment pipeline builds a docker image and pushes it to the registry."
COFFEE_NOTE = "Espresso extraction works best around ninety two degrees celsius."


def _ingest(client, content: str, title: str | None = None) -> str:
    payload = {"source_type": "note", "content": content}
    if title:
        payload["title"] = title
    response = client.post("/ingest", json=payload)
    assert response.status_code == 201
    return response.json()["item"]["id"]


# --- empty corpus ---------------------------------------------------------- #
def test_query_on_an_empty_store_explains_itself_without_calling_the_model(client, fake_openai):
    response = client.post("/query", json={"question": "What did I save?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == EMPTY_STORE_ANSWER
    assert body["sources"] == []
    assert body["retrieved_chunk_count"] == 0
    assert fake_openai.chat_calls == []
    assert fake_openai.embed_calls == []


# --- retrieval ------------------------------------------------------------- #
def test_query_returns_an_answer_with_cited_sources(client, fake_openai):
    item_id = _ingest(client, LOCKS_NOTE, title="Advisory locks")
    fake_openai.answer = "They are session scoped [1]."

    response = client.post("/query", json={"question": "Are advisory locks session scoped?"})

    assert response.status_code == 200
    body = response.json()
    assert body["question"] == "Are advisory locks session scoped?"
    assert body["answer"] == "They are session scoped [1]."
    assert body["model"] == "gpt-4o-mini"
    assert body["retrieved_chunk_count"] == 1

    source = body["sources"][0]
    assert source["citation"] == 1
    assert source["item_id"] == item_id
    assert source["title"] == "Advisory locks"
    assert source["source_type"] == "note"
    assert source["chunk_index"] == 0
    assert source["snippet"] == LOCKS_NOTE
    assert 0.0 <= source["score"] <= 1.0


def test_the_most_relevant_chunk_is_cited_first(client):
    _ingest(client, COFFEE_NOTE, title="coffee")
    _ingest(client, DEPLOY_NOTE, title="deploys")
    _ingest(client, LOCKS_NOTE, title="locks")

    body = client.post("/query", json={"question": "advisory locks session rollback postgres"}).json()

    assert body["sources"][0]["title"] == "locks"
    scores = [source["score"] for source in body["sources"]]
    assert scores == sorted(scores, reverse=True)


def test_citations_are_numbered_from_one_and_match_the_prompt(client, fake_openai):
    _ingest(client, LOCKS_NOTE, title="locks")
    _ingest(client, DEPLOY_NOTE, title="deploys")
    _ingest(client, COFFEE_NOTE, title="coffee")

    body = client.post("/query", json={"question": "postgres locks"}).json()

    citations = [source["citation"] for source in body["sources"]]
    assert citations == list(range(1, len(citations) + 1))

    prompt = fake_openai.last_user_prompt
    for source in body["sources"]:
        assert f"[{source['citation']}]" in prompt


def test_top_k_defaults_to_the_configured_value(client):
    for index in range(6):
        _ingest(client, f"Note number {index} about databases and locks and queries.")

    body = client.post("/query", json={"question": "databases"}).json()

    assert body["retrieved_chunk_count"] == 4
    assert len(body["sources"]) == 4


def test_top_k_can_be_overridden_per_request(client):
    for index in range(6):
        _ingest(client, f"Note number {index} about databases and locks and queries.")

    body = client.post("/query", json={"question": "databases", "top_k": 2}).json()

    assert body["retrieved_chunk_count"] == 2
    assert len(body["sources"]) == 2


def test_top_k_larger_than_the_corpus_is_clamped(client):
    _ingest(client, LOCKS_NOTE)

    body = client.post("/query", json={"question": "locks", "top_k": 20}).json()

    assert body["retrieved_chunk_count"] == 1


def test_retrieval_can_span_chunks_of_one_long_item(client):
    long_note = (LOCKS_NOTE + " ") * 60
    _ingest(client, long_note, title="long")

    body = client.post("/query", json={"question": "advisory locks"}).json()

    assert body["retrieved_chunk_count"] > 1
    assert {source["chunk_index"] for source in body["sources"]} != {0}


# --- prompt construction --------------------------------------------------- #
def test_the_prompt_carries_the_question_and_numbered_context(client, fake_openai):
    _ingest(client, LOCKS_NOTE)

    client.post("/query", json={"question": "Do locks survive rollback?"})

    system_prompt, user_prompt = fake_openai.chat_calls[0]
    assert "only facts present in the context" in system_prompt.lower()
    assert "cite" in system_prompt.lower()
    assert "Question: Do locks survive rollback?" in user_prompt
    assert f"[1] {LOCKS_NOTE}" in user_prompt


def test_long_snippets_are_truncated_in_the_response(client):
    _ingest(client, "word " * 400, title="wordy")

    snippet = client.post("/query", json={"question": "word"}).json()["sources"][0]["snippet"]

    assert len(snippet) <= 403
    assert snippet.endswith("...")


# --- validation ------------------------------------------------------------ #
@pytest.mark.parametrize("question", ["", "   ", "\n"])
def test_a_blank_question_is_rejected(client, question):
    response = client.post("/query", json={"question": question})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_a_missing_question_is_rejected(client):
    response = client.post("/query", json={})

    assert response.status_code == 422
    fields = {problem["field"] for problem in response.json()["error"]["details"]["fields"]}
    assert "question" in fields


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_out_of_range_top_k_is_rejected(client, top_k):
    _ingest(client, LOCKS_NOTE)

    response = client.post("/query", json={"question": "locks", "top_k": top_k})

    assert response.status_code == 422


# --- provider failures ----------------------------------------------------- #
def test_a_failing_completion_surfaces_as_502(client, fake_openai):
    _ingest(client, LOCKS_NOTE)
    fake_openai.chat_error = connection_error()

    response = client.post("/query", json={"question": "locks"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_unavailable"


def test_a_rate_limited_question_embedding_surfaces_as_429(client, fake_openai):
    _ingest(client, LOCKS_NOTE)
    fake_openai.embed_error = rate_limit_error()

    response = client.post("/query", json={"question": "locks"})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "ai_rate_limited"


def test_a_missing_api_key_surfaces_as_503(client, fake_openai):
    from app.clients.openai_client import MissingCredentialsError

    _ingest(client, LOCKS_NOTE)
    fake_openai.chat_error = MissingCredentialsError("OPENAI_API_KEY is not set.")

    response = client.post("/query", json={"question": "locks"})

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "ai_not_configured"
