"""Test doubles. The suite never touches the network.

``FakeOpenAIClient`` embeds text with a deterministic hashing trick instead of a
real model. Texts that share words end up with similar vectors, so cosine
similarity behaves sensibly and retrieval assertions stay meaningful.
"""

from __future__ import annotations

import re
from zlib import crc32

import httpx
import openai

from app.clients.openai_client import ModelResolution, OpenAIClientError, _map_sdk_error

EMBEDDING_DIM = 64
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _raise_like_the_real_client(error: Exception | None) -> None:
    """Mirror the real client: raw SDK errors are normalised before they escape.

    Tests can therefore inject realistic ``openai.*`` exceptions and still see
    the internal taxonomy the HTTP layer is built around.
    """
    if error is None:
        return
    if isinstance(error, OpenAIClientError):
        raise error
    raise _map_sdk_error(error) from error


def fake_embedding(text: str) -> list[float]:
    """Bag-of-words vector using a stable hash (``hash()`` is salted per process)."""
    vector = [0.0] * EMBEDDING_DIM
    for token in _TOKEN_PATTERN.findall(text.lower()):
        vector[crc32(token.encode()) % EMBEDDING_DIM] += 1.0
    return vector


class FakeOpenAIClient:
    """Drop-in replacement for :class:`app.clients.openai_client.OpenAIClient`."""

    def __init__(
        self,
        chat_model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        answer: str = "Advisory locks are session scoped [1].",
        embed_error: Exception | None = None,
        chat_error: Exception | None = None,
    ) -> None:
        self._chat_model = chat_model
        self._embedding_model = embedding_model
        self.answer = answer
        self.embed_error = embed_error
        self.chat_error = chat_error
        self.embed_calls: list[list[str]] = []
        self.chat_calls: list[tuple[str, str]] = []

    @property
    def chat_model(self) -> str:
        return self._chat_model

    @property
    def embedding_model(self) -> str:
        return self._embedding_model

    def resolve_chat_model(self) -> ModelResolution:
        return ModelResolution(requested=self._chat_model, resolved=self._chat_model, available=True)

    def embed_texts(self, texts) -> list[list[float]]:
        _raise_like_the_real_client(self.embed_error)
        texts = list(texts)
        self.embed_calls.append(texts)
        return [fake_embedding(text) for text in texts]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def complete_chat(self, system_prompt: str, user_prompt: str) -> str:
        _raise_like_the_real_client(self.chat_error)
        self.chat_calls.append((system_prompt, user_prompt))
        return self.answer

    @property
    def last_user_prompt(self) -> str:
        return self.chat_calls[-1][1]


# --------------------------------------------------------------------------- #
# Stubs for the raw OpenAI SDK (used to test the real client wrapper)
# --------------------------------------------------------------------------- #
class _StubModels:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.retrieved: list[str] = []

    def retrieve(self, model_id: str):
        self.retrieved.append(model_id)
        if self.error:
            raise self.error
        return {"id": model_id, "object": "model"}


class _StubEmbeddingDatum:
    def __init__(self, index: int, embedding: list[float]) -> None:
        self.index = index
        self.embedding = embedding


class _StubEmbeddingResponse:
    def __init__(self, data) -> None:
        self.data = data


class _StubEmbeddings:
    def __init__(self, error: Exception | None = None, shuffle: bool = False, drop_one: bool = False) -> None:
        self.error = error
        self.shuffle = shuffle
        self.drop_one = drop_one
        self.calls: list[dict] = []

    def create(self, *, model: str, input):  # noqa: A002 - mirrors the SDK signature
        self.calls.append({"model": model, "input": input})
        if self.error:
            raise self.error
        data = [_StubEmbeddingDatum(index, fake_embedding(text)) for index, text in enumerate(input)]
        if self.drop_one:
            data = data[:-1]
        if self.shuffle:
            data = list(reversed(data))
        return _StubEmbeddingResponse(data)


class _StubMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _StubChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _StubMessage(content)


class _StubChatResponse:
    def __init__(self, choices) -> None:
        self.choices = choices


_UNSET = object()


class _StubCompletions:
    def __init__(
        self,
        content: str | None = "answer",
        error: Exception | None = None,
        choices=None,
        reject_temperature: bool = False,
    ) -> None:
        self.content = content
        self.error = error
        self._choices = choices
        self.reject_temperature = reject_temperature
        self.calls: list[dict] = []

    def create(self, *, model: str, messages, temperature=_UNSET):
        sent_temperature = temperature is not _UNSET
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "sent_temperature": sent_temperature,
                "temperature": temperature if sent_temperature else None,
            }
        )
        if self.reject_temperature and sent_temperature:
            raise temperature_rejected_error()
        if self.error:
            raise self.error
        if self._choices is not None:
            return _StubChatResponse(self._choices)
        return _StubChatResponse([_StubChoice(self.content)])


class _StubChat:
    def __init__(self, completions: _StubCompletions) -> None:
        self.completions = completions


class StubSdk:
    """Minimal stand-in for ``openai.OpenAI``."""

    def __init__(
        self,
        retrieve_error: Exception | None = None,
        embed_error: Exception | None = None,
        chat_error: Exception | None = None,
        chat_content: str | None = "answer",
        chat_choices=None,
        shuffle_embeddings: bool = False,
        drop_one_embedding: bool = False,
        reject_temperature: bool = False,
    ) -> None:
        self.models = _StubModels(retrieve_error)
        self.embeddings = _StubEmbeddings(embed_error, shuffle_embeddings, drop_one_embedding)
        self.chat = _StubChat(_StubCompletions(chat_content, chat_error, chat_choices, reject_temperature))


# --------------------------------------------------------------------------- #
# Real SDK exceptions, cheap to construct
# --------------------------------------------------------------------------- #
def _response(status_code: int, url: str = "https://api.openai.com/v1/models/x") -> httpx.Response:
    return httpx.Response(status_code, request=httpx.Request("GET", url))


def model_not_found_error() -> openai.NotFoundError:
    return openai.NotFoundError("The model does not exist", response=_response(404), body=None)


def auth_error() -> openai.AuthenticationError:
    return openai.AuthenticationError("Invalid API key", response=_response(401), body=None)


def rate_limit_error() -> openai.RateLimitError:
    return openai.RateLimitError("Rate limit reached", response=_response(429), body=None)


def connection_error() -> openai.APIConnectionError:
    return openai.APIConnectionError(request=httpx.Request("GET", "https://api.openai.com/v1/models"))


def server_error() -> openai.InternalServerError:
    return openai.InternalServerError("upstream boom", response=_response(500), body=None)


def temperature_rejected_error() -> openai.BadRequestError:
    """Mirrors the real 400 returned by gpt-5.6-* models for an explicit temperature."""
    return openai.BadRequestError(
        "Error code: 400 - Unsupported value: 'temperature' does not support 0.2 with this "
        "model. Only the default (1) value is supported.",
        response=_response(400),
        body=None,
    )
