"""Tasks 5 and 6: ingesting notes and URLs, and listing what was saved."""

from __future__ import annotations

import pytest

from app.services import ingestion as ingestion_module
from app.services.url_extractor import ExtractedPage, UrlExtractionError
from tests.fakes import auth_error, rate_limit_error

LONG_NOTE = (
    "Advisory locks in Postgres are held for the duration of the session, not the transaction. "
    "That means a rollback does not release them. " * 40
)


@pytest.fixture
def stub_extract(monkeypatch):
    """Replace the network fetch with a controllable stub."""

    calls: list[dict] = []

    def _install(page: ExtractedPage | None = None, error: Exception | None = None):
        def fake_fetch(url: str, timeout_seconds: float, max_chars: int) -> ExtractedPage:
            calls.append({"url": url, "timeout_seconds": timeout_seconds, "max_chars": max_chars})
            if error:
                raise error
            return page or ExtractedPage(url=url, title="Extracted Title", text="Extracted article body.")

        monkeypatch.setattr(ingestion_module, "fetch_and_extract", fake_fetch)
        return calls

    return _install


# --- notes ----------------------------------------------------------------- #
def test_ingesting_a_note_returns_201_and_a_summary(client, fake_openai):
    response = client.post("/ingest", json={"source_type": "note", "content": "Advisory locks are session scoped."})

    assert response.status_code == 201
    body = response.json()
    assert body["item"]["source_type"] == "note"
    assert body["item"]["title"] == "Advisory locks are session scoped."
    assert body["item"]["source_url"] is None
    assert body["item"]["preview"] == "Advisory locks are session scoped."
    assert body["chunk_count"] == 1
    assert body["item"]["chunk_count"] == 1
    # The note text was embedded exactly once.
    assert fake_openai.embed_calls == [["Advisory locks are session scoped."]]


def test_an_explicit_title_wins(client):
    response = client.post(
        "/ingest",
        json={"source_type": "note", "content": "some body text", "title": "My chosen title"},
    )

    assert response.json()["item"]["title"] == "My chosen title"


def test_a_long_note_title_is_truncated_on_a_word_boundary(client):
    content = "This first line is deliberately much longer than the seventy character title limit allows"
    title = client.post("/ingest", json={"source_type": "note", "content": content}).json()["item"]["title"]

    assert title.endswith("...")
    assert len(title) <= 74
    assert "charac" not in title.replace("...", "") or title.replace("...", "").split()[-1] != "charac"


def test_a_long_note_is_split_into_several_chunks(client, fake_openai):
    response = client.post("/ingest", json={"source_type": "note", "content": LONG_NOTE})

    body = response.json()
    assert body["chunk_count"] > 1
    assert len(fake_openai.embed_calls[0]) == body["chunk_count"]
    assert body["item"]["char_count"] > 900


def test_note_preview_is_truncated_but_full_text_is_kept(client):
    body = client.post("/ingest", json={"source_type": "note", "content": LONG_NOTE}).json()

    assert body["item"]["preview"].endswith("...")
    assert len(body["item"]["preview"]) <= 244
    assert body["item"]["char_count"] > len(body["item"]["preview"])


# --- notes: validation ----------------------------------------------------- #
@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_blank_note_content_is_rejected(client, content):
    response = client.post("/ingest", json={"source_type": "note", "content": content})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_unknown_source_type_is_rejected(client):
    response = client.post("/ingest", json={"source_type": "podcast", "content": "hello"})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_missing_fields_are_reported_per_field(client):
    response = client.post("/ingest", json={})

    assert response.status_code == 422
    fields = {problem["field"] for problem in response.json()["error"]["details"]["fields"]}
    assert {"source_type", "content"} <= fields


def test_nothing_is_stored_when_validation_fails(client):
    client.post("/ingest", json={"source_type": "note", "content": "  "})

    assert client.get("/items").json()["count"] == 0


# --- urls ------------------------------------------------------------------ #
def test_ingesting_a_url_extracts_title_and_body(client, stub_extract):
    calls = stub_extract(
        ExtractedPage(url="https://example.com/post", title="Real Title", text="The article body text.")
    )

    response = client.post("/ingest", json={"source_type": "url", "content": "https://example.com/post"})

    assert response.status_code == 201
    body = response.json()
    assert body["item"]["source_type"] == "url"
    assert body["item"]["title"] == "Real Title"
    assert body["item"]["source_url"] == "https://example.com/post"
    assert body["item"]["preview"] == "The article body text."
    assert calls[0]["url"] == "https://example.com/post"


def test_url_ingestion_passes_configured_limits(client, stub_extract):
    calls = stub_extract()

    client.post("/ingest", json={"source_type": "url", "content": "https://example.com"})

    assert calls[0]["timeout_seconds"] == 15.0
    assert calls[0]["max_chars"] == 200_000


def test_url_without_a_title_falls_back_to_the_hostname(client, stub_extract):
    stub_extract(ExtractedPage(url="https://blog.example.com/x", title=None, text="Body."))

    body = client.post("/ingest", json={"source_type": "url", "content": "https://blog.example.com/x"}).json()

    assert body["item"]["title"] == "blog.example.com"


@pytest.mark.parametrize("url", ["not-a-url", "ftp://example.com/file", "https://", "javascript:alert(1)"])
def test_malformed_urls_are_rejected_with_422(client, url):
    response = client.post("/ingest", json={"source_type": "url", "content": url})

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_an_unreachable_url_returns_502(client, stub_extract):
    stub_extract(error=UrlExtractionError("Could not reach the URL: connection refused."))

    response = client.post("/ingest", json={"source_type": "url", "content": "https://example.com/gone"})

    assert response.status_code == 502
    error = response.json()["error"]
    assert error["code"] == "upstream_fetch_failed"
    assert "Could not reach" in error["message"]
    assert error["details"]["url"] == "https://example.com/gone"


def test_a_page_with_no_article_text_returns_502(client, stub_extract):
    stub_extract(error=UrlExtractionError("no readable article text was found"))

    response = client.post("/ingest", json={"source_type": "url", "content": "https://example.com/spa"})

    assert response.status_code == 502
    assert "readable article text" in response.json()["error"]["message"]


# --- provider failures ----------------------------------------------------- #
def test_embedding_rate_limit_surfaces_as_429(client, fake_openai):
    fake_openai.embed_error = rate_limit_error()

    response = client.post("/ingest", json={"source_type": "note", "content": "hello"})

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "ai_rate_limited"


def test_embedding_auth_failure_surfaces_as_502(client, fake_openai):
    fake_openai.embed_error = auth_error()

    response = client.post("/ingest", json={"source_type": "note", "content": "hello"})

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ai_authentication_failed"


def test_a_failed_embedding_stores_nothing(client, fake_openai):
    fake_openai.embed_error = rate_limit_error()
    client.post("/ingest", json={"source_type": "note", "content": "hello"})

    fake_openai.embed_error = None
    assert client.get("/items").json()["count"] == 0


# --- GET /items ------------------------------------------------------------ #
def test_items_is_empty_before_anything_is_saved(client):
    response = client.get("/items")

    assert response.status_code == 200
    assert response.json() == {"items": [], "count": 0}


def test_items_returns_newest_first(client, stub_extract):
    stub_extract(ExtractedPage(url="https://example.com/a", title="From the web", text="Web body."))
    client.post("/ingest", json={"source_type": "note", "content": "First note"})
    client.post("/ingest", json={"source_type": "note", "content": "Second note"})
    client.post("/ingest", json={"source_type": "url", "content": "https://example.com/a"})

    body = client.get("/items").json()

    assert body["count"] == 3
    assert [item["title"] for item in body["items"]] == ["From the web", "Second note", "First note"]


def test_item_summaries_expose_metadata_without_full_content(client):
    client.post("/ingest", json={"source_type": "note", "content": LONG_NOTE})

    item = client.get("/items").json()["items"][0]

    assert set(item) == {
        "id",
        "source_type",
        "title",
        "source_url",
        "created_at",
        "chunk_count",
        "char_count",
        "preview",
    }
    assert "content" not in item
    assert item["chunk_count"] > 1
