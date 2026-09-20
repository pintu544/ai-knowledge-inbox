"""Thin wrapper around the OpenAI SDK.

Every outbound AI call goes through this module so that:

* provider-specific exceptions are mapped to a small internal error taxonomy the
  HTTP layer can translate into sensible status codes;
* the SDK is trivial to mock in tests (no network calls in the test suite);
* the configured chat model is resolved once at startup rather than discovered
  mid-request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import openai

from app.config import DEFAULT_CHAT_MODEL, Settings

logger = logging.getLogger(__name__)


class OpenAIClientError(RuntimeError):
    """Base class for provider failures surfaced to the application."""


class MissingCredentialsError(OpenAIClientError):
    """No API key configured."""


class AuthenticationFailedError(OpenAIClientError):
    """The API key was rejected."""


class RateLimitedError(OpenAIClientError):
    """Provider rate limit or quota exceeded."""


class ProviderUnavailableError(OpenAIClientError):
    """Network problem, timeout, or provider-side error."""


class InvalidRequestError(OpenAIClientError):
    """The provider rejected the request (bad model name, oversized input, ...)."""


@dataclass(frozen=True)
class ModelResolution:
    """Outcome of validating the configured chat model at startup."""

    requested: str
    resolved: str
    available: bool
    detail: str | None = None

    @property
    def used_fallback(self) -> bool:
        return self.requested != self.resolved


def _is_temperature_rejection(error: InvalidRequestError) -> bool:
    """True when the provider refused the request specifically over temperature."""
    message = str(error).lower()
    return "temperature" in message and ("unsupported" in message or "does not support" in message)


def _map_sdk_error(error: Exception) -> OpenAIClientError:
    """Translate an SDK exception into our internal taxonomy."""
    if isinstance(error, openai.AuthenticationError):
        return AuthenticationFailedError("OpenAI rejected the configured API key.")
    if isinstance(error, openai.PermissionDeniedError):
        return AuthenticationFailedError("The configured API key is not permitted to use this resource.")
    if isinstance(error, openai.RateLimitError):
        return RateLimitedError("OpenAI rate limit or quota exceeded. Retry shortly.")
    if isinstance(error, (openai.APITimeoutError, openai.APIConnectionError)):
        return ProviderUnavailableError("Could not reach OpenAI (network error or timeout).")
    if isinstance(error, openai.NotFoundError):
        return InvalidRequestError("OpenAI could not find the requested resource (check the model name).")
    if isinstance(error, openai.BadRequestError):
        return InvalidRequestError(f"OpenAI rejected the request: {error}")
    if isinstance(error, openai.APIStatusError):
        return ProviderUnavailableError(f"OpenAI returned an error response (HTTP {error.status_code}).")
    if isinstance(error, openai.OpenAIError):
        return OpenAIClientError(f"OpenAI client error: {error}")
    return ProviderUnavailableError(f"Unexpected error calling OpenAI: {error}")


class OpenAIClient:
    """Application-facing AI client.

    The underlying SDK object is created lazily so the app can boot (and serve
    ``/health``) even when no API key is configured yet.
    """

    def __init__(self, settings: Settings, sdk: Any | None = None) -> None:
        self._settings = settings
        self._sdk = sdk
        self._chat_model = settings.openai_model
        self._embedding_model = settings.openai_embedding_model

    # --- configuration -----------------------------------------------------
    @property
    def chat_model(self) -> str:
        """Chat model actually in use (may differ from config after fallback)."""
        return self._chat_model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    @property
    def sdk(self) -> Any:
        if self._sdk is None:
            if not self._settings.has_openai_credentials:
                raise MissingCredentialsError(
                    "OPENAI_API_KEY is not set. Add it to your .env file to enable ingestion and queries."
                )
            self._sdk = openai.OpenAI(
                api_key=self._settings.openai_api_key,
                timeout=self._settings.openai_timeout_seconds,
            )
        return self._sdk

    # --- startup validation ------------------------------------------------
    def resolve_chat_model(self) -> ModelResolution:
        """Check the configured chat model and fall back if it is unreachable.

        Uses ``GET /v1/models/{model}``, which is the authoritative answer for
        *this* API key. On any failure we keep serving with
        :data:`~app.config.DEFAULT_CHAT_MODEL` instead of letting every later
        ``/query`` blow up.
        """
        requested = self._settings.openai_model

        if not self._settings.has_openai_credentials:
            self._chat_model = requested
            return ModelResolution(
                requested=requested,
                resolved=requested,
                available=False,
                detail="OPENAI_API_KEY is not set, so the model could not be validated.",
            )

        try:
            self.sdk.models.retrieve(requested)
        except Exception as error:  # noqa: BLE001 - startup must never crash the app
            mapped = _map_sdk_error(error)
            if requested == DEFAULT_CHAT_MODEL:
                self._chat_model = requested
                return ModelResolution(
                    requested=requested,
                    resolved=requested,
                    available=False,
                    detail=f"{mapped} Keeping the configured model because it is already the default.",
                )
            self._chat_model = DEFAULT_CHAT_MODEL
            return ModelResolution(
                requested=requested,
                resolved=DEFAULT_CHAT_MODEL,
                available=False,
                detail=str(mapped),
            )

        self._chat_model = requested
        return ModelResolution(requested=requested, resolved=requested, available=True)

    # --- embeddings --------------------------------------------------------
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        if not texts:
            return []

        try:
            response = self.sdk.embeddings.create(model=self._embedding_model, input=list(texts))
        except OpenAIClientError:
            raise
        except Exception as error:  # noqa: BLE001 - normalised below
            raise _map_sdk_error(error) from error

        # The API documents order preservation, but sorting by index is cheap
        # insurance against a mismatch between chunks and their vectors.
        ordered = sorted(response.data, key=lambda datum: datum.index)
        vectors = [list(datum.embedding) for datum in ordered]

        if len(vectors) != len(texts):
            raise ProviderUnavailableError(
                f"Embedding count mismatch: asked for {len(texts)} vectors, received {len(vectors)}."
            )

        logger.debug("embedded texts", extra={"count": len(vectors), "model": self._embedding_model})
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text (convenience wrapper around :meth:`embed_texts`)."""
        return self.embed_texts([text])[0]

    # --- chat --------------------------------------------------------------
    def complete_chat(self, system_prompt: str, user_prompt: str) -> str:
        """Run a single-turn chat completion and return the message text.

        ``temperature`` is only sent when ``OPENAI_TEMPERATURE`` is configured.
        Newer models accept just their default value and reject anything else
        with HTTP 400, so if a configured value is refused we log it and retry
        once without the parameter rather than failing the user's query.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        temperature = self._settings.openai_temperature

        try:
            response = self._create_completion(messages, temperature)
        except InvalidRequestError as error:
            if temperature is None or not _is_temperature_rejection(error):
                raise
            logger.warning(
                "model rejected the configured temperature; retrying with the model default",
                extra={"model": self._chat_model, "temperature": temperature},
            )
            response = self._create_completion(messages, None)

        if not response.choices:
            raise ProviderUnavailableError("OpenAI returned no choices for the chat completion.")

        content = response.choices[0].message.content
        if not content or not content.strip():
            raise ProviderUnavailableError("OpenAI returned an empty answer.")

        return content.strip()

    def _create_completion(self, messages: list[dict[str, str]], temperature: float | None) -> Any:
        """Issue the completion call, omitting ``temperature`` when unset."""
        payload: dict[str, Any] = {"model": self._chat_model, "messages": messages}
        if temperature is not None:
            payload["temperature"] = temperature

        try:
            return self.sdk.chat.completions.create(**payload)
        except OpenAIClientError:
            raise
        except Exception as error:  # noqa: BLE001 - normalised below
            raise _map_sdk_error(error) from error
