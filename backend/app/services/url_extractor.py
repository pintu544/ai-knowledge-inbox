"""Server-side URL fetching and readability-style content extraction.

Uses trafilatura to pull the main article body and title, discarding navigation,
ads, and boilerplate. That matters for RAG: chunks made from a raw HTML-to-text
dump are polluted with menu labels and cookie notices, which drags retrieval
quality down and produces ugly citations.

Fetching is wrapped in a hard timeout and a size ceiling, because a URL is
untrusted input: it may hang, redirect somewhere huge, or serve a binary.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)

_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0 Safari/537.36 AIKnowledgeInbox/0.1"
)
_TEXTUAL_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain", "application/xml", "text/xml")

#: Below this, extraction has effectively failed. trafilatura's fallback will
#: happily return a nav label like "Menu" for a page with no article, and
#: ingesting that pollutes retrieval with meaningless chunks. Notes have no such
#: floor: a short note is a deliberate choice by the user.
MIN_EXTRACTED_CHARS = 80

#: Inline markup survives extraction inside code blocks on some sites (styled
#: terminal output on docs pages is a common culprit). Left in, it wastes prompt
#: tokens, muddies embeddings, and makes citations unreadable.
#:
#: Deliberately an explicit tag list rather than ``<[^>]+>``: prose legitimately
#: contains angle brackets (a note about a ``<script>`` tag, ``a < b``), and a
#: greedy pattern would silently delete the user's words.
_MARKUP_TAGS = (
    "span|font|div|p|a|b|i|u|em|strong|code|pre|br|hr|img|small|sup|sub|mark|"
    "table|thead|tbody|tfoot|tr|td|th|ul|ol|li|dl|dt|dd|h[1-6]|"
    "section|article|nav|header|footer|aside|main|figure|figcaption|blockquote|label"
)
_HTML_TAG = re.compile(rf"</?(?:{_MARKUP_TAGS})(?:\s[^>]{{0,400}})?/?>", re.IGNORECASE)

#: Documentation generators append a pilcrow to headings as an anchor link.
_TITLE_NOISE = "¶#"


class UrlExtractionError(RuntimeError):
    """The URL could not be fetched, or held no readable article text."""


@dataclass(frozen=True)
class ExtractedPage:
    """Readable content pulled from a web page."""

    url: str
    title: str | None
    text: str


def validate_url(url: str) -> str:
    """Normalise and sanity-check a URL before any network call.

    Raises:
        ValueError: for anything that is not an absolute http(s) URL.
    """
    candidate = url.strip()
    parsed = urlparse(candidate)

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")
    if not parsed.netloc:
        raise ValueError("URL is missing a hostname")

    return candidate


def fetch_and_extract(url: str, timeout_seconds: float = 15.0, max_chars: int = 200_000) -> ExtractedPage:
    """Fetch ``url`` and return its main article text.

    Raises:
        ValueError: the URL is not a usable http(s) URL.
        UrlExtractionError: the fetch failed, the response was not textual, or no
            article text could be recovered.
    """
    target = validate_url(url)
    html = _fetch_html(target, timeout_seconds)

    extracted = trafilatura.extract(
        html,
        url=target,
        favor_precision=True,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    text = clean_extracted_text(extracted or "")
    if len(text) < MIN_EXTRACTED_CHARS:
        raise UrlExtractionError(
            "The page was fetched but no readable article text was found "
            f"(recovered {len(text)} characters, need at least {MIN_EXTRACTED_CHARS}). "
            "It may be a JavaScript-rendered app, a paywall, or a non-article page."
        )

    if len(text) > max_chars:
        logger.warning(
            "truncating extracted page",
            extra={"url": target, "original_chars": len(text), "max_chars": max_chars},
        )
        text = text[:max_chars]

    title = _extract_title(html, target)
    logger.info("extracted url", extra={"url": target, "chars": len(text), "has_title": bool(title)})
    return ExtractedPage(url=target, title=title, text=text)


def clean_extracted_text(text: str) -> str:
    """Remove leftover inline markup, decode entities, drop anchor pilcrows.

    The pilcrow is dropped because documentation generators attach one to every
    heading as an anchor link, and it shows up verbatim in citation snippets.
    """
    without_markup = _HTML_TAG.sub("", text)
    return html.unescape(without_markup).replace("¶", "").strip()


def _fetch_html(url: str, timeout_seconds: float) -> str:
    """GET the URL, following redirects, and return the body as text."""
    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={"User-Agent": _BROWSER_USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.TimeoutException as error:
        raise UrlExtractionError(f"Timed out after {timeout_seconds:g}s fetching the URL.") from error
    except httpx.HTTPStatusError as error:
        raise UrlExtractionError(
            f"The site returned HTTP {error.response.status_code} for that URL."
        ) from error
    except httpx.RequestError as error:
        raise UrlExtractionError(f"Could not reach the URL: {error}.") from error

    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith(_TEXTUAL_CONTENT_TYPES):
        raise UrlExtractionError(
            f"Unsupported content type '{content_type}'. Only HTML and plain-text pages can be ingested."
        )

    return response.text


def _extract_title(html: str, url: str) -> str | None:
    """Best-effort page title from trafilatura's metadata."""
    try:
        metadata = trafilatura.extract_metadata(html, default_url=url)
    except Exception as error:  # noqa: BLE001 - metadata is a nice-to-have, never fatal
        logger.debug("title extraction failed", extra={"url": url, "detail": str(error)})
        return None

    title = getattr(metadata, "title", None) if metadata else None
    if not title:
        return None

    cleaned = clean_extracted_text(title).strip(_TITLE_NOISE).strip()
    return cleaned or None
