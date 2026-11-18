# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Structured JSON logging with structlog, PII scrubbing, OTel trace injection, and ring buffer."""

from __future__ import annotations

import collections
from collections.abc import Mapping
import copy
import enum
import json
import logging
import os
import re
import sys
import threading
from typing import Any

import structlog

from thrilled_to_announce.observability.tracing import get_current_trace_ids

DEFAULT_LOGGER_NAME = "thrilled_to_announce"
DEFAULT_LOG_BUFFER_MAXLEN = 1000

_HEX32_RE = re.compile(r"^[0-9a-f]{32}$")
_HEX16_RE = re.compile(r"^[0-9a-f]{16}$")
_SAFE_PHASES = frozenset({"INTENT", "OUTCOME"})
_SAFE_LEVELS = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "exception"}
)
_SENSITIVE_DICT_KEYS = frozenset({
    "api_key",
    "apikey",
    "secret",
    "secret_key",
    "client_secret",
    "access_token",
    "auth_token",
    "bearer_token",
    "authorization",
    "password",
    "passwd",
})

_FALLBACK_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_FALLBACK_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_FALLBACK_PHONE_RE = re.compile(
    r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)
_FALLBACK_SALARY_RE = re.compile(r"\$\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b")

_TELEMETRY_LOCK = threading.Lock()
_TELEMETRY_RING_BUFFER: collections.deque[dict[str, Any]] = collections.deque(
    maxlen=DEFAULT_LOG_BUFFER_MAXLEN
)

_CONFIG_LOCK = threading.Lock()
_LOGGING_CONFIGURED = False
_STDERR_HANDLER: logging.StreamHandler[Any] | None = None
_CLOUD_LOGGING_HANDLER: logging.Handler | None = None
_ATTACHED_CLOUD_LOGGING_CLIENTS: set[tuple[Any, str | None]] = set()
_SCRUB_REENTRANCY = threading.local()


class _DynamicStderrHandler(logging.StreamHandler[Any]):
    """StreamHandler that resolves ``sys.stderr`` dynamically at emit time."""

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stderr
        super().emit(record)


def _fallback_regex_scrub(text: str) -> str:
    """Emergency local regex scrubber used only if PiiScrubber is re-entered."""
    out = _FALLBACK_EMAIL_RE.sub("[REDACTED_EMAIL]", text)
    out = _FALLBACK_SSN_RE.sub("[REDACTED_SSN]", out)
    out = _FALLBACK_PHONE_RE.sub("[REDACTED_PHONE]", out)
    return _FALLBACK_SALARY_RE.sub("[REDACTED_SALARY]", out)


def _scrub_string_with_pii_scrubber(
    text: str, *, use_fallback_only: bool = False
) -> str:
    """Scrub sensitive tokens from a string via PiiScrubber."""
    if not text:
        return text
    if use_fallback_only:
        return _fallback_regex_scrub(text)
    try:
        from thrilled_to_announce.observability.pii_scrubber import scrub_text

        return scrub_text(text).scrubbed_text
    except Exception:
        return _fallback_regex_scrub(text)


def _scrub_nested_value(
    value: Any,
    *,
    key_hint: str | None = None,
    use_fallback_only: bool = False,
) -> Any:
    """Recursively scrub all string values while preserving valid structural IDs."""
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if (
        key_hint is not None
        and key_hint.lower() in _SENSITIVE_DICT_KEYS
        and isinstance(value, str)
        and value.strip()
        and value.strip() != "[REDACTED_SECRET]"
    ):
        return "[REDACTED_SECRET]"
    if isinstance(value, str):
        if key_hint == "trace_id" and _HEX32_RE.match(value):
            return value
        if key_hint in {"span_id", "parent_span_id"} and _HEX16_RE.match(value):
            return value
        if key_hint == "phase" and value in _SAFE_PHASES:
            return value
        if key_hint == "level" and value.lower() in _SAFE_LEVELS:
            return value
        if key_hint == "timestamp":
            return value
        return _scrub_string_with_pii_scrubber(
            value, use_fallback_only=use_fallback_only
        )
    if isinstance(value, enum.Enum):
        return _scrub_nested_value(
            value.value,
            key_hint=key_hint,
            use_fallback_only=use_fallback_only,
        )
    if isinstance(value, Mapping):
        return {
            str(k): _scrub_nested_value(
                v,
                key_hint=str(k),
                use_fallback_only=use_fallback_only,
            )
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _scrub_nested_value(item, use_fallback_only=use_fallback_only)
            for item in value
        ]
    return _scrub_string_with_pii_scrubber(
        str(value), use_fallback_only=use_fallback_only
    )


def _inject_trace_context_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Inject active W3C ``trace_id`` and ``span_id`` when not already set."""
    del logger, method_name
    trace_id, span_id = get_current_trace_ids()
    if not event_dict.get("trace_id"):
        event_dict["trace_id"] = trace_id
    if not event_dict.get("span_id"):
        event_dict["span_id"] = span_id
    return event_dict


def _pii_scrubber_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Scrub PII from all event fields prior to ring-buffer storage and JSON output."""
    del logger, method_name
    reentered = bool(getattr(_SCRUB_REENTRANCY, "active", False))
    _SCRUB_REENTRANCY.active = True
    try:
        for key in list(event_dict.keys()):
            if key in {"_record", "_from_structlog"}:
                continue
            event_dict[key] = _scrub_nested_value(
                event_dict[key],
                key_hint=key,
                use_fallback_only=reentered,
            )
        return event_dict
    finally:
        if not reentered:
            _SCRUB_REENTRANCY.active = False


def _ring_buffer_processor(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    """Capture structured telemetry dictionary into the thread-safe ring buffer."""
    del logger, method_name
    snapshot = {
        k: v
        for k, v in event_dict.items()
        if k not in {"_record", "_from_structlog"}
    }
    json_ready_entry: dict[str, Any] = json.loads(
        json.dumps(snapshot, default=str)
    )
    with _TELEMETRY_LOCK:
        _TELEMETRY_RING_BUFFER.append(json_ready_entry)
    return event_dict


def _try_build_cloud_logging_handler(
    project_id: str | None = None,
) -> logging.Handler | None:
    """Build an optional Google Cloud Logging handler with safe local fallback."""
    resolved_project = project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    if resolved_project and resolved_project.strip().lower() in {
        "",
        "your_project_id",
        "<your_project_id>",
        "your-project-id",
        "placeholder",
        "none",
        "null",
    }:
        resolved_project = None
    try:
        import google.cloud.logging as cloud_logging  # type: ignore[import-not-found]

        client_key = (cloud_logging.Client, resolved_project)
        if client_key in _ATTACHED_CLOUD_LOGGING_CLIENTS:
            return None
        client = (
            cloud_logging.Client(project=resolved_project)
            if resolved_project
            else cloud_logging.Client()
        )
        handler = client.get_default_handler()
        if isinstance(handler, logging.Handler):
            _ATTACHED_CLOUD_LOGGING_CLIENTS.add(client_key)
            return handler
        return None
    except ImportError:
        return None
    except Exception:
        return None


def configure_logging(
    level: int | str | None = None,
    enable_cloud_logging: bool | None = None,
    project_id: str | None = None,
) -> None:
    """Configure structlog JSON rendering, PII scrubbing, trace injection, and handlers."""
    global _LOGGING_CONFIGURED, _STDERR_HANDLER, _CLOUD_LOGGING_HANDLER

    raw_level = level if level is not None else os.getenv("LOG_LEVEL", "INFO")
    resolved_level = (
        raw_level
        if isinstance(raw_level, int)
        else getattr(logging, str(raw_level).upper(), logging.INFO)
    )
    cloud_flag = (
        enable_cloud_logging
        if enable_cloud_logging is not None
        else os.getenv("ENABLE_CLOUD_LOGGING", "").strip().lower()
        in {"1", "true", "yes"}
    )

    with _CONFIG_LOCK:
        shared_processors: list[Any] = [
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _inject_trace_context_processor,
            _pii_scrubber_processor,
        ]
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.JSONRenderer(),
            ],
        )

        root_logger = logging.getLogger()
        if _STDERR_HANDLER is None:
            _STDERR_HANDLER = _DynamicStderrHandler()
        _STDERR_HANDLER.setFormatter(formatter)
        if _STDERR_HANDLER not in root_logger.handlers:
            root_logger.addHandler(_STDERR_HANDLER)
        if root_logger.level == logging.NOTSET or root_logger.level > resolved_level:
            root_logger.setLevel(resolved_level)

        if cloud_flag:
            cloud_handler = _try_build_cloud_logging_handler(project_id=project_id)
            if cloud_handler is not None:
                cloud_handler.setFormatter(formatter)
                root_logger.addHandler(cloud_handler)
                _CLOUD_LOGGING_HANDLER = cloud_handler

        for noisy_logger_name in ("httpx", "httpcore", "urllib3", "aiosqlite"):
            logging.getLogger(noisy_logger_name).setLevel(logging.WARNING)

        structlog.configure(
            processors=[
                *shared_processors,
                _ring_buffer_processor,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=False,
        )
        _LOGGING_CONFIGURED = True


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog BoundLogger emitting scrubbed JSON records."""
    if not _LOGGING_CONFIGURED:
        configure_logging()
    stdlib_logger = logging.getLogger(name)
    if stdlib_logger.getEffectiveLevel() > logging.INFO:
        stdlib_logger.setLevel(logging.INFO)
    return structlog.stdlib.get_logger(name)


def get_telemetry_logs(limit: int = 200) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent scrubbed structured JSON log entries."""
    if limit <= 0:
        return []
    with _TELEMETRY_LOCK:
        recent = list(_TELEMETRY_RING_BUFFER)[-limit:]
    return [copy.deepcopy(entry) for entry in recent]


def clear_telemetry_logs() -> None:
    """Clear all entries from the in-memory telemetry ring buffer."""
    with _TELEMETRY_LOCK:
        _TELEMETRY_RING_BUFFER.clear()
