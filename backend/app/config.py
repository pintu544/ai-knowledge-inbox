"""Application configuration.

Everything is read from environment variables, optionally seeded by a ``.env``
file at the repository root (see ``backend/.env.example``). Real environment
variables always take precedence over ``.env`` values.

Model names are deliberately configuration-driven: pointing the app at a
different chat model is a one-line ``.env`` change and requires no code change.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = APP_DIR.parent
REPO_ROOT = BACKEND_ROOT.parent

#: Verified, widely available defaults. Used when the configured model cannot be
#: reached so the app degrades to something that works instead of failing every
#: request.
DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class Settings(BaseSettings):
    """Typed application settings."""

    model_config = SettingsConfigDict(
        # Later files win, so a backend-local .env can override the repo-root one.
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = DEFAULT_CHAT_MODEL
    openai_embedding_model: str = DEFAULT_EMBEDDING_MODEL
    openai_timeout_seconds: float = 30.0
    #: Left unset on purpose: newer models (e.g. gpt-5.6-*) only accept their
    #: default temperature and reject an explicit value with HTTP 400. Omitting
    #: the parameter keeps the app compatible across model generations.
    openai_temperature: float | None = None

    # --- RAG tunables ---
    chunk_size: int = 900
    chunk_overlap: int = 150
    retrieval_top_k: int = 4

    # --- Ingestion limits ---
    url_fetch_timeout_seconds: float = 15.0
    max_content_chars: int = 200_000

    # --- Ops ---
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @field_validator("openai_model", mode="before")
    @classmethod
    def _chat_model_or_default(cls, value: object) -> object:
        """An empty/whitespace value means "not configured", so use the default."""
        if not isinstance(value, str) or not value.strip():
            return DEFAULT_CHAT_MODEL
        return value.strip()

    @field_validator("openai_embedding_model", mode="before")
    @classmethod
    def _embedding_model_or_default(cls, value: object) -> object:
        if not isinstance(value, str) or not value.strip():
            return DEFAULT_EMBEDDING_MODEL
        return value.strip()

    @field_validator("openai_temperature", mode="before")
    @classmethod
    def _optional_temperature(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("openai_temperature")
    @classmethod
    def _temperature_in_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 2.0:
            raise ValueError("OPENAI_TEMPERATURE must be between 0 and 2")
        return value

    @field_validator("chunk_size")
    @classmethod
    def _positive_chunk_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def _non_negative_overlap(cls, value: int) -> int:
        if value < 0:
            raise ValueError("CHUNK_OVERLAP must be 0 or greater")
        return value

    @field_validator("retrieval_top_k")
    @classmethod
    def _positive_top_k(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("RETRIEVAL_TOP_K must be greater than 0")
        return value

    @model_validator(mode="after")
    def _overlap_smaller_than_chunk(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE to make forward progress")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        """CORS origins as a list. ``*`` disables the allow-list."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def has_openai_credentials(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor (import-safe, used as a FastAPI dependency)."""
    return Settings()
