"""Error taxonomy and a single consistent JSON error envelope.

Every failure the API returns looks like::

    {"error": {"code": "invalid_input", "message": "...", "details": {...}}}

so the frontend has exactly one shape to parse.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.clients.openai_client import (
    AuthenticationFailedError,
    InvalidRequestError,
    MissingCredentialsError,
    OpenAIClientError,
    ProviderUnavailableError,
    RateLimitedError,
)

logger = logging.getLogger(__name__)


class ApiError(Exception):
    """An application error that maps directly onto an HTTP response."""

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details = details or {}


class InvalidInputError(ApiError):
    """Caller-supplied data is unusable (HTTP 422)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status.HTTP_422_UNPROCESSABLE_ENTITY, "invalid_input", message, details)


class UpstreamFetchError(ApiError):
    """A URL could not be fetched or contained no usable content (HTTP 502)."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status.HTTP_502_BAD_GATEWAY, "upstream_fetch_failed", message, details)


class AiProviderError(ApiError):
    """The AI provider failed (HTTP 502/429/503 depending on cause)."""

    def __init__(self, status_code: int, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(status_code, code, message, details)


def _envelope(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"error": {"code": code, "message": message}}
    if details:
        payload["error"]["details"] = details
    return payload


def api_error_to_response(error: ApiError) -> JSONResponse:
    return JSONResponse(status_code=error.status_code, content=_envelope(error.code, error.message, error.details))


def translate_provider_error(error: OpenAIClientError) -> AiProviderError:
    """Map an OpenAI client failure onto an HTTP-shaped application error."""
    if isinstance(error, MissingCredentialsError):
        return AiProviderError(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "ai_not_configured",
            str(error),
        )
    if isinstance(error, AuthenticationFailedError):
        return AiProviderError(
            status.HTTP_502_BAD_GATEWAY,
            "ai_authentication_failed",
            str(error),
        )
    if isinstance(error, RateLimitedError):
        return AiProviderError(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "ai_rate_limited",
            str(error),
        )
    if isinstance(error, InvalidRequestError):
        return AiProviderError(
            status.HTTP_502_BAD_GATEWAY,
            "ai_request_rejected",
            str(error),
        )
    if isinstance(error, ProviderUnavailableError):
        return AiProviderError(
            status.HTTP_502_BAD_GATEWAY,
            "ai_unavailable",
            str(error),
        )
    return AiProviderError(status.HTTP_502_BAD_GATEWAY, "ai_error", str(error))


def register_exception_handlers(app: FastAPI) -> None:
    """Attach handlers so *all* error responses share one envelope."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, error: ApiError) -> JSONResponse:
        logger.warning(
            "request failed",
            extra={"error_code": error.code, "status_code": error.status_code, "detail": error.message},
        )
        return api_error_to_response(error)

    @app.exception_handler(OpenAIClientError)
    async def _handle_provider_error(_: Request, error: OpenAIClientError) -> JSONResponse:
        translated = translate_provider_error(error)
        logger.error(
            "ai provider call failed",
            extra={"error_code": translated.code, "status_code": translated.status_code, "detail": translated.message},
        )
        return api_error_to_response(translated)

    @app.exception_handler(RequestValidationError)
    async def _handle_validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        fields = [
            {
                "field": ".".join(str(part) for part in problem.get("loc", []) if part != "body"),
                "message": problem.get("msg", "Invalid value."),
            }
            for problem in error.errors()
        ]
        logger.info("request validation failed", extra={"fields": fields})
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=_envelope("invalid_input", "Request body failed validation.", {"fields": fields}),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(_: Request, error: StarletteHTTPException) -> JSONResponse:
        code = "not_found" if error.status_code == status.HTTP_404_NOT_FOUND else "http_error"
        return JSONResponse(
            status_code=error.status_code,
            content=_envelope(code, str(error.detail)),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, error: Exception) -> JSONResponse:
        logger.exception("unhandled server error", extra={"error_type": type(error).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("internal_error", "Something went wrong on the server."),
        )
