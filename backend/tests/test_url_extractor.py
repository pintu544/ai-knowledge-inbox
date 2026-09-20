"""Task 5: URL validation, fetching, and readability-style extraction.

``httpx`` is driven through a MockTransport, so no real requests are made.
"""

from __future__ import annotations

import httpx
import pytest

from app.services import url_extractor
from app.services.url_extractor import UrlExtractionError, fetch_and_extract, validate_url

ARTICLE_HTML = """
<!doctype html>
<html>
  <head><title>Advisory Locks Explained</title></head>
  <body>
    <nav><ul><li>Home</li><li>Blog</li><li>Contact us today</li></ul></nav>
    <article>
      <h1>Advisory Locks Explained</h1>
      <p>Postgres advisory locks are held for the lifetime of the session rather than
      the transaction. A rollback therefore does not release them, which surprises
      people who expect transactional cleanup semantics to apply here.</p>
      <p>Use pg_advisory_unlock to release one explicitly, or close the session and
      let the server reclaim every lock the session was holding at that moment.</p>
    </article>
    <footer>Copyright 2026. Subscribe to our newsletter for more updates.</footer>
  </body>
</html>
"""


@pytest.fixture
def mock_http(monkeypatch):
    """Point ``url_extractor`` at a scripted transport instead of the network."""

    # Captured before patching: url_extractor.httpx is the global module, so the
    # factory would otherwise call itself.
    real_client_cls = httpx.Client

    def _install(handler):
        def client_factory(*_args, **kwargs):
            return real_client_cls(transport=httpx.MockTransport(handler), follow_redirects=True)

        monkeypatch.setattr(url_extractor.httpx, "Client", client_factory)

    return _install


def _respond(status_code: int = 200, html: str = ARTICLE_HTML, content_type: str = "text/html; charset=utf-8"):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, text=html, headers={"content-type": content_type})

    return handler


# --- validation ------------------------------------------------------------ #
@pytest.mark.parametrize(
    "url",
    ["https://example.com", "http://example.com/path?q=1", "  https://example.com/trailing  "],
)
def test_valid_urls_are_accepted_and_trimmed(url):
    assert validate_url(url) == url.strip()


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("example.com", "http"),
        ("ftp://example.com", "http"),
        ("javascript:alert(1)", "http"),
        ("https://", "hostname"),
        ("", "http"),
    ],
)
def test_invalid_urls_are_rejected(url, message):
    with pytest.raises(ValueError, match=message):
        validate_url(url)


# --- extraction ------------------------------------------------------------ #
def test_extracts_article_text_and_drops_boilerplate(mock_http):
    mock_http(_respond())

    page = fetch_and_extract("https://example.com/post")

    assert page.url == "https://example.com/post"
    assert page.title == "Advisory Locks Explained"
    assert "advisory locks are held for the lifetime of the session" in page.text.lower()
    # Navigation and footer boilerplate should not make it into the chunks.
    assert "Contact us today" not in page.text
    assert "newsletter" not in page.text


def test_extraction_truncates_at_the_configured_ceiling(mock_http):
    body = "<p>" + ("Sentence about locks and sessions. " * 400) + "</p>"
    mock_http(_respond(html=f"<html><body><article>{body}</article></body></html>"))

    page = fetch_and_extract("https://example.com/long", max_chars=500)

    assert len(page.text) == 500


def test_residual_inline_markup_is_stripped(mock_http):
    """Docs sites leak styled spans out of code blocks; they must not reach chunks."""
    body = (
        "<p>Run the server and watch the output carefully before continuing with the rest of this guide.</p>"
        '<pre><code><span style="background-color:#007166"><font color="#D3D7CF"> INFO </font></span> '
        "Uvicorn running on port 8000</code></pre>"
    )
    mock_http(_respond(html=f"<html><head><title>Guide</title></head><body><article>{body}</article></body></html>"))

    page = fetch_and_extract("https://example.com/docs")

    assert "<span" not in page.text
    assert "background-color" not in page.text
    assert "INFO" in page.text
    assert "Uvicorn running on port 8000" in page.text


def test_html_entities_are_decoded(mock_http):
    body = "<p>Use a &lt;script&gt; tag &amp; then reload the page to confirm the behaviour is correct.</p>"
    mock_http(_respond(html=f"<html><head><title>T</title></head><body><article>{body}</article></body></html>"))

    page = fetch_and_extract("https://example.com/entities")

    assert "&amp;" not in page.text
    assert "<script>" in page.text


def test_anchor_pilcrows_are_trimmed_from_titles(mock_http):
    """Docs generators append a pilcrow anchor to headings."""
    html_page = ARTICLE_HTML.replace("<title>Advisory Locks Explained</title>", "<title>First Steps¶</title>").replace(
        "<h1>Advisory Locks Explained</h1>", "<h1>First Steps¶</h1>"
    )
    mock_http(_respond(html=html_page))

    page = fetch_and_extract("https://example.com/first-steps")

    assert page.title == "First Steps"
    # Body text carries the same anchor markers; they should not reach citations.
    assert "¶" not in page.text


def test_prose_containing_angle_brackets_is_preserved(mock_http):
    """The markup stripper must not eat the user's actual words."""
    body = "<p>The condition a &lt; b holds, and the &lt;script&gt; tag belongs in the document head instead.</p>"
    mock_http(_respond(html=f"<html><head><title>T</title></head><body><article>{body}</article></body></html>"))

    page = fetch_and_extract("https://example.com/brackets")

    assert "a < b" in page.text
    assert "<script>" in page.text


def test_a_page_without_a_title_still_extracts(mock_http):
    html = "<html><body><article><p>" + ("Body sentence about locks. " * 20) + "</p></article></body></html>"
    mock_http(_respond(html=html))

    page = fetch_and_extract("https://example.com/untitled")

    assert page.text
    assert page.title is None or isinstance(page.title, str)


# --- failure modes --------------------------------------------------------- #
@pytest.mark.parametrize("status_code", [404, 403, 500, 503])
def test_error_statuses_are_reported(mock_http, status_code):
    mock_http(_respond(status_code=status_code))

    with pytest.raises(UrlExtractionError, match=str(status_code)):
        fetch_and_extract("https://example.com/missing")


def test_a_timeout_is_reported_clearly(mock_http):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    mock_http(handler)

    with pytest.raises(UrlExtractionError, match="Timed out"):
        fetch_and_extract("https://example.com/slow", timeout_seconds=2)


def test_a_connection_failure_is_reported_clearly(mock_http):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("name resolution failed", request=request)

    mock_http(handler)

    with pytest.raises(UrlExtractionError, match="Could not reach"):
        fetch_and_extract("https://nope.invalid/x")


def test_binary_content_types_are_refused(mock_http):
    mock_http(_respond(html="%PDF-1.7 binary", content_type="application/pdf"))

    with pytest.raises(UrlExtractionError, match="Unsupported content type"):
        fetch_and_extract("https://example.com/paper.pdf")


def test_a_page_with_no_article_text_is_reported(mock_http):
    """trafilatura's fallback will return a nav label; too short to be an article."""
    mock_http(_respond(html="<html><body><nav>Menu</nav></body></html>"))

    with pytest.raises(UrlExtractionError, match="no readable article text"):
        fetch_and_extract("https://example.com/empty")


def test_a_nearly_empty_page_is_reported_with_the_recovered_length(mock_http):
    mock_http(_respond(html="<html><body><article><p>Too short.</p></article></body></html>"))

    with pytest.raises(UrlExtractionError, match="need at least 80"):
        fetch_and_extract("https://example.com/thin")


def test_invalid_urls_never_reach_the_network(mock_http):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("the fetch should not have been attempted")

    mock_http(handler)

    with pytest.raises(ValueError):
        fetch_and_extract("not-a-url")
