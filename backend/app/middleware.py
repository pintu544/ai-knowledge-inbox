"""HTTP middleware.

A minimal, dependency-free rate limiter. The app is intentionally
unauthenticated (single-user by design), so this is the one guard that keeps a
public deployment from turning into unbounded OpenAI spend if the URL leaks.

It is a fixed-window counter keyed by client IP, kept in process memory. That is
enough for a single instance; a multi-instance deployment would move the counter
to a shared store (e.g. Redis) — see the README tradeoffs.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

_WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Fixed-window per-IP rate limit. Disabled when ``limit_per_minute`` is 0."""

    def __init__(self, app, limit_per_minute: int) -> None:
        super().__init__(app)
        self._limit = limit_per_minute
        self._lock = Lock()
        #: client ip -> (window_start_epoch, count)
        self._hits: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    async def dispatch(self, request: Request, call_next):
        if self._limit <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()

        with self._lock:
            window_start, count = self._hits[client_ip]
            if now - window_start >= _WINDOW_SECONDS:
                window_start, count = now, 0
            count += 1
            self._hits[client_ip] = (window_start, count)
            over_limit = count > self._limit
            retry_after = max(1, int(_WINDOW_SECONDS - (now - window_start)))

        if over_limit:
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={
                    "error": {
                        "code": "rate_limited",
                        "message": "Too many requests. Please slow down and try again shortly.",
                    }
                },
            )

        return await call_next(request)
