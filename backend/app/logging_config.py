"""Structured logging setup.

Logs are emitted as one JSON object per line so they can be shipped to any log
aggregator without a custom parser. Set ``LOG_JSON=false`` for human-readable
output while developing.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

#: Attributes present on every ``LogRecord``. Anything outside this set was
#: supplied by the caller via ``extra={...}`` and is therefore worth emitting.
_RESERVED_RECORD_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_HUMAN_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class JsonLogFormatter(logging.Formatter):
    """Render log records as single-line JSON, including ``extra`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)

        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", as_json: bool = True) -> None:
    """Install a single stdout handler on the root logger.

    Idempotent: repeated calls replace existing handlers rather than stacking
    them, which matters because uvicorn may import the app more than once.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonLogFormatter() if as_json else logging.Formatter(_HUMAN_FORMAT))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    # Let uvicorn's records flow through our formatter instead of its own.
    for uvicorn_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(uvicorn_logger)
        logger.handlers.clear()
        logger.propagate = True

    # trafilatura is chatty about pages it cannot parse; we report those ourselves.
    logging.getLogger("trafilatura").setLevel(logging.ERROR)
