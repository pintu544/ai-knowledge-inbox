"""Task 4: the OpenAI wrapper shapes requests correctly and normalises errors.

No test in this file touches the network; the SDK is stubbed.
"""

from __future__ import annotations

import pytest

from app.clients.openai_client import (
    AuthenticationFailedError,
    InvalidRequestError,
    MissingCredentialsError,
    OpenAIClient,
    ProviderUnavailableError,
    RateLimitedError,
)
from app.config import Settings
from tests.fakes import (
    StubSdk,
    auth_error,
    connection_error,
    model_not_found_error,
    rate_limit_error,
    server_error,
)


def _client(sdk: StubSdk | None = None, **env_overrides) -> OpenAIClient:
    return OpenAIClient(Settings(**env_overrides), sdk=sdk or StubSdk())


# --- embeddings ------------------------------------------------------------ #
def test_embed_texts_sends_the_configured_model_and_batches_input():
    sdk = StubSdk()
    client = _client(sdk)

    vectors = client.embed_texts(["alpha", "beta"])

    assert sdk.embeddings.calls == [{"model": "text-embedding-3-small", "input": ["alpha", "beta"]}]
    assert len(vectors) == 2
    assert all(isinstance(value, float) for value in vectors[0])


def test_embed_texts_short_circuits_on_empty_input():
    sdk = StubSdk()

    assert _client(sdk).embed_texts([]) == []
    assert sdk.embeddings.calls == []


def test_embed_texts_restores_order_if_the_provider_returns_them_shuffled():
    ordered = _client(StubSdk()).embed_texts(["alpha", "beta"])
    shuffled = _client(StubSdk(shuffle_embeddings=True)).embed_texts(["alpha", "beta"])

    assert shuffled == ordered


def test_embed_texts_rejects_a_count_mismatch():
    client = _client(StubSdk(drop_one_embedding=True))

    with pytest.raises(ProviderUnavailableError, match="mismatch"):
        client.embed_texts(["alpha", "beta"])


def test_embed_text_returns_a_single_vector():
    vector = _client().embed_text("alpha")

    assert isinstance(vector, list)
    assert isinstance(vector[0], float)


# --- chat ------------------------------------------------------------------ #
def test_temperature_is_omitted_by_default():
    """Newer models reject an explicit temperature, so we must not send one."""
    sdk = StubSdk(chat_content="hello")
    client = _client(sdk)

    assert client.complete_chat("system", "user") == "hello"
    assert sdk.chat.completions.calls[0]["sent_temperature"] is False


def test_configured_temperature_is_sent(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    sdk = StubSdk(chat_content="hello")

    _client(sdk).complete_chat("system", "user")

    assert sdk.chat.completions.calls[0]["temperature"] == 0.2


def test_rejected_temperature_is_retried_without_it(monkeypatch):
    """gpt-5.6-* answer fine at their default temperature; don't fail the query."""
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-luna")
    sdk = StubSdk(chat_content="recovered", reject_temperature=True)

    answer = _client(sdk).complete_chat("system", "user")

    assert answer == "recovered"
    assert [call["sent_temperature"] for call in sdk.chat.completions.calls] == [True, False]


def test_other_bad_requests_are_not_retried(monkeypatch):
    monkeypatch.setenv("OPENAI_TEMPERATURE", "0.2")
    sdk = StubSdk(chat_error=model_not_found_error())
    client = _client(sdk)

    with pytest.raises(InvalidRequestError):
        client.complete_chat("system", "user")

    assert len(sdk.chat.completions.calls) == 1


def test_chat_sends_system_and_user_messages_in_order():
    sdk = StubSdk()

    _client(sdk).complete_chat("be terse", "what is up?")

    assert sdk.chat.completions.calls[0]["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is up?"},
    ]


@pytest.mark.parametrize("content", [None, "", "   "])
def test_empty_answers_are_treated_as_provider_failures(content):
    client = _client(StubSdk(chat_content=content))

    with pytest.raises(ProviderUnavailableError, match="empty"):
        client.complete_chat("system", "user")


def test_missing_choices_is_a_provider_failure():
    client = _client(StubSdk(chat_choices=[]))

    with pytest.raises(ProviderUnavailableError, match="no choices"):
        client.complete_chat("system", "user")


# --- error mapping --------------------------------------------------------- #
@pytest.mark.parametrize(
    ("sdk_error", "expected"),
    [
        (auth_error(), AuthenticationFailedError),
        (rate_limit_error(), RateLimitedError),
        (connection_error(), ProviderUnavailableError),
        (model_not_found_error(), InvalidRequestError),
        (server_error(), ProviderUnavailableError),
    ],
)
def test_sdk_errors_map_to_internal_taxonomy(sdk_error, expected):
    client = _client(StubSdk(embed_error=sdk_error))

    with pytest.raises(expected):
        client.embed_texts(["alpha"])


def test_missing_credentials_are_reported_before_any_call(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    client = OpenAIClient(Settings())

    with pytest.raises(MissingCredentialsError, match="OPENAI_API_KEY"):
        client.embed_texts(["alpha"])
