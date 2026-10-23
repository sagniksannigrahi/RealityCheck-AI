"""Shared API dependencies: authentication, rate limiting, error type."""

from __future__ import annotations

import hashlib
import hmac
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Request

from app.core.config import get_settings


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, details=None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True)
class Principal:
    owner_id: str
    authenticated: bool


def _owner_for_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def get_principal(request: Request) -> Principal:
    s = get_settings()
    supplied = request.headers.get("x-api-key")
    auth = request.headers.get("authorization", "")
    if not supplied and auth.lower().startswith("bearer "):
        supplied = auth[7:].strip()
    keys = s.api_key_list
    if supplied:
        for k in keys:
            if hmac.compare_digest(k.encode(), supplied.encode()):
                return Principal(owner_id=_owner_for_key(k), authenticated=True)
        if s.auth_required or keys:
            raise APIError(401, "invalid_api_key", "The API key is not valid.")
    if s.auth_required:
        raise APIError(401, "authentication_required", "An API key is required (X-API-Key header).")
    return Principal(owner_id="local", authenticated=False)


class RateLimiter:
    """Sliding-window limiter per principal (in-process; use a gateway for multi-instance deployments)."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_s: float = 60.0) -> None:
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._hits[key]
            while q and now - q[0] > window_s:
                q.popleft()
            if len(q) >= limit:
                retry = int(window_s - (now - q[0])) + 1
                raise APIError(429, "rate_limited", f"Too many requests; retry in {retry}s.", {"retry_after_s": retry})
            q.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()
