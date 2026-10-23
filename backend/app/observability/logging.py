"""Structured JSON logging with request/job correlation IDs and secret redaction."""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
job_id_var: ContextVar[str | None] = ContextVar("job_id", default=None)

# Patterns for values that must never reach logs or API error messages.
_SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"), "sk-ant-[REDACTED]"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "sk-[REDACTED]"),
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-]{8,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)((?:api[_-]?key|secret|password|token)\s*[=:]\s*)['\"]?[^\s'\",;]{4,}"), r"\1[REDACTED]"),
    (re.compile(r"(?i)(x-api-key['\"]?\s*[:=]\s*['\"]?)[^\s'\",;]{4,}"), r"\1[REDACTED]"),
    (re.compile(r"(postgres(?:ql)?(?:\+\w+)?://[^:/\s]+:)[^@\s]+(@)"), r"\1[REDACTED]\2"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AKIA[REDACTED]"),
]


def redact(text: str) -> str:
    """Remove secrets (API keys, bearer tokens, DB passwords) from free text."""
    if not text:
        return text
    for pattern, repl in _SECRET_PATTERNS:
        text = pattern.sub(repl, text)
    return text


_extra_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Register a literal secret value (e.g. a configured API key) for exact-match redaction."""
    if value and len(value) >= 6:
        _extra_secrets.add(value)


def redact_all(text: str) -> str:
    text = redact(text)
    for s in _extra_secrets:
        if s in text:
            text = text.replace(s, "[REDACTED]")
    return text


_RESERVED = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_all(record.getMessage()),
        }
        rid = request_id_var.get()
        jid = job_id_var.get()
        if rid:
            payload["request_id"] = rid
        if jid:
            payload["job_id"] = jid
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value if isinstance(value, int | float | bool | type(None)) else redact_all(
                    str(value)
                )
        if record.exc_info:
            payload["exc"] = redact_all(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        rid = request_id_var.get()
        jid = job_id_var.get()
        ctx = " ".join(x for x in [f"rid={rid}" if rid else "", f"job={jid}" if jid else ""] if x)
        return redact_all(f"{base} {ctx}".rstrip())


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        JsonFormatter() if json_logs else TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level.upper())
    # Quieten noisy third-party loggers.
    for name in ("urllib3", "httpx", "PIL", "ultralytics", "transformers", "multipart"):
        logging.getLogger(name).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
