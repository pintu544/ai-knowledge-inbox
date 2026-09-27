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
import ipaddress
import logging
import re
import socket
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import trafilatura

logger = logging.getLogger(__name__)

#: How many redirects we are willing to follow. Each hop is re-validated.
_MAX_REDIRECTS = 5

#: Read the body in bounded chunks so a huge (or compressed-bomb) response is
#: stopped mid-stream instead of being fully buffered into memory.
_STREAM_CHUNK_BYTES = 64 * 1024

#: Hard cap on how many raw bytes we will read from a response body. Set well
#: above ``max_chars`` so extraction still has the full article to work with,
#: but low enough that a runaway response cannot exhaust memory.
_MAX_RESPONSE_BYTES = 10 * 1024 * 1024

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


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True for any address a public fetch has no business reaching.

    Covers loopback (127.0.0.0/8, ::1), private ranges (10/8, 172.16/12,
    192.168/16, fc00::/7), link-local (169.254/16 — including the cloud metadata
    endpoint 169.254.169.254 — and fe80::/10), and other reserved/unspecified
    space. IPv4-mapped IPv6 addresses are unwrapped so ``::ffff:127.0.0.1`` is
    caught too.
    """
    if getattr(ip, "ipv4_mapped", None) is not None:
        ip = ip.ipv4_mapped  # type: ignore[assignment]

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _assert_public_host(host: str) -> None:
    """Resolve ``host`` and reject it if any resolved IP is non-public.

    This is the core SSRF guard: it runs before every request and after every
    redirect, so a URL cannot reach localhost, a private network, or the cloud
    metadata service — even via a redirect or a hostname that resolves to one.

    Raises:
        UrlExtractionError: the host cannot be resolved or points somewhere
            internal.
    """
    # A bare IP literal in the URL is checked directly (no DNS needed).
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None

    if literal is not None:
        if _is_blocked_ip(literal):
            raise UrlExtractionError("That URL points to a private or reserved address and cannot be fetched.")
        return

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise UrlExtractionError(f"Could not resolve the host '{host}'.") from error

    for info in infos:
        sockaddr = info[4]
        try:
            resolved = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        if _is_blocked_ip(resolved):
            raise UrlExtractionError("That URL resolves to a private or reserved address and cannot be fetched.")


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
    """GET the URL and return the body as text.

    Redirects are followed manually so every hop is re-validated against the
    SSRF guard (httpx's built-in redirect following would skip that check). The
    body is streamed and capped so an oversized or compressed-bomb response is
    stopped mid-read rather than fully buffered.
    """
    headers = {"User-Agent": _BROWSER_USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*"}

    try:
        with httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=False,  # we follow (and re-validate) manually
            headers=headers,
        ) as client:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                _assert_public_host(urlparse(current).hostname or "")

                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise UrlExtractionError("The site returned a redirect with no destination.")
                        # Resolve relative redirects against the current URL, then
                        # re-validate on the next loop iteration.
                        current = str(response.url.join(location))
                        continue

                    response.raise_for_status()
                    _reject_declared_oversize(response)
                    _reject_non_textual(response)
                    return _read_capped_text(response)

            raise UrlExtractionError(f"The URL redirected more than {_MAX_REDIRECTS} times; giving up.")
    except httpx.TimeoutException as error:
        raise UrlExtractionError(f"Timed out after {timeout_seconds:g}s fetching the URL.") from error
    except httpx.HTTPStatusError as error:
        raise UrlExtractionError(
            f"The site returned HTTP {error.response.status_code} for that URL."
        ) from error
    except httpx.RequestError as error:
        raise UrlExtractionError(f"Could not reach the URL: {error}.") from error


def _reject_declared_oversize(response: httpx.Response) -> None:
    """Fail fast when Content-Length already declares an oversized body."""
    declared = response.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > _MAX_RESPONSE_BYTES:
        raise UrlExtractionError(
            f"The page is too large to ingest (declared {int(declared)} bytes, "
            f"limit {_MAX_RESPONSE_BYTES})."
        )


def _reject_non_textual(response: httpx.Response) -> None:
    content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith(_TEXTUAL_CONTENT_TYPES):
        raise UrlExtractionError(
            f"Unsupported content type '{content_type}'. Only HTML and plain-text pages can be ingested."
        )


def _read_capped_text(response: httpx.Response) -> str:
    """Stream the body, stopping once the byte cap is exceeded, then decode."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes(_STREAM_CHUNK_BYTES):
        total += len(chunk)
        if total > _MAX_RESPONSE_BYTES:
            raise UrlExtractionError(
                f"The page exceeded the {_MAX_RESPONSE_BYTES}-byte fetch limit while downloading."
            )
        chunks.append(chunk)

    body = b"".join(chunks)
    encoding = response.encoding or "utf-8"
    try:
        return body.decode(encoding, errors="replace")
    except (LookupError, ValueError):
        return body.decode("utf-8", errors="replace")


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
