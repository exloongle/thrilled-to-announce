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
"""OpenTelemetry distributed tracing with Cloud Trace and in-memory span export."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
import contextvars
import enum
import json
import os
import re
import threading
from typing import Any
import uuid

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, Tracer

DEFAULT_TRACER_NAME = "thrilled_to_announce"
DEFAULT_MAX_RECORDED_SPANS = 1000

_SENSITIVE_ATTR_KEYS = frozenset({
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

_TRACING_LOCK = threading.Lock()
_PROVIDER: TracerProvider | None = None
_IN_MEMORY_EXPORTER: InMemorySpanExporter | None = None
_IN_MEMORY_ATTACHED = False
_ATTACHED_CLOUD_EXPORTERS: set[tuple[Any, str | None]] = set()
_FALLBACK_TRACE_CTX: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("thrilled_fallback_trace_ctx", default=None)
)


def _create_in_memory_exporter(
    max_spans: int = DEFAULT_MAX_RECORDED_SPANS,
) -> InMemorySpanExporter:
    """Instantiate InMemorySpanExporter across OpenTelemetry SDK versions."""
    try:
        return InMemorySpanExporter(max_spans=max_spans)
    except TypeError:
        return InMemorySpanExporter()


def _scrub_string(value: str) -> str:
    """Scrub PII from span string values using PiiScrubber with safe fallback."""
    if not value:
        return value
    try:
        from thrilled_to_announce.observability.pii_scrubber import scrub_text

        return scrub_text(value).scrubbed_text
    except Exception:
        out = _FALLBACK_EMAIL_RE.sub("[REDACTED_EMAIL]", value)
        out = _FALLBACK_SSN_RE.sub("[REDACTED_SSN]", out)
        out = _FALLBACK_PHONE_RE.sub("[REDACTED_PHONE]", out)
        return _FALLBACK_SALARY_RE.sub("[REDACTED_SALARY]", out)


def _sanitize_span_attributes(
    attributes: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Normalize and PII-scrub span attributes into valid OpenTelemetry types."""
    if not attributes:
        return {}
    cleaned: dict[str, Any] = {}
    for raw_key, raw_val in attributes.items():
        if raw_val is None:
            continue
        key = str(raw_key)
        if key.lower() in _SENSITIVE_ATTR_KEYS and isinstance(raw_val, str) and raw_val.strip():
            cleaned[key] = "[REDACTED_SECRET]"
        elif isinstance(raw_val, (bool, int, float)):
            cleaned[key] = raw_val
        elif isinstance(raw_val, str):
            cleaned[key] = _scrub_string(raw_val)
        elif isinstance(raw_val, enum.Enum):
            cleaned[key] = _scrub_string(str(raw_val.value))
        elif isinstance(raw_val, Sequence) and not isinstance(
            raw_val, (bytes, bytearray)
        ):
            seq_items: list[str | bool | int | float] = []
            for item in raw_val:
                if isinstance(item, (bool, int, float)):
                    seq_items.append(item)
                elif item is not None:
                    seq_items.append(_scrub_string(str(item)))
            cleaned[key] = tuple(seq_items)
        else:
            cleaned[key] = _scrub_string(json.dumps(raw_val, default=str))
    return cleaned


def _ensure_tracer_provider() -> TracerProvider:
    """Initialize or adopt the global SDK TracerProvider idempotently."""
    global _PROVIDER, _IN_MEMORY_EXPORTER, _IN_MEMORY_ATTACHED
    with _TRACING_LOCK:
        if _IN_MEMORY_EXPORTER is None:
            _IN_MEMORY_EXPORTER = _create_in_memory_exporter(
                DEFAULT_MAX_RECORDED_SPANS
            )
        current = trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            _PROVIDER = current
        elif _PROVIDER is None:
            service_name = os.getenv("OTEL_SERVICE_NAME", "thrilled-to-announce")
            resource = Resource.create({"service.name": service_name})
            _PROVIDER = TracerProvider(resource=resource)
            trace.set_tracer_provider(_PROVIDER)
        if not _IN_MEMORY_ATTACHED and _PROVIDER is not None:
            _PROVIDER.add_span_processor(SimpleSpanProcessor(_IN_MEMORY_EXPORTER))
            _IN_MEMORY_ATTACHED = True
        assert _PROVIDER is not None
        return _PROVIDER


def configure_tracing(
    enable_cloud_export: bool = False,
    project_id: str | None = None,
) -> Tracer:
    """Configure OpenTelemetry TracerProvider with in-memory and optional Cloud Trace export.

    Idempotent across repeated calls: sets the global TracerProvider only once
    to avoid OpenTelemetry override warnings, and attaches CloudTraceSpanExporter
    via BatchSpanProcessor when ``enable_cloud_export=True`` and available.
    """
    provider = _ensure_tracer_provider()
    env_cloud_flag = os.getenv("OTEL_EXPORT_TO_CLOUD", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if enable_cloud_export or env_cloud_flag:
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
        with _TRACING_LOCK:
            try:
                from opentelemetry.exporter.cloud_trace import (  # type: ignore[import-not-found]
                    CloudTraceSpanExporter,
                )

                exporter_key = (CloudTraceSpanExporter, resolved_project)
                if exporter_key not in _ATTACHED_CLOUD_EXPORTERS:
                    cloud_exporter = (
                        CloudTraceSpanExporter(project_id=resolved_project)
                        if resolved_project
                        else CloudTraceSpanExporter()
                    )
                    provider.add_span_processor(BatchSpanProcessor(cloud_exporter))
                    _ATTACHED_CLOUD_EXPORTERS.add(exporter_key)
            except ImportError:
                pass
            except Exception:
                pass
    return provider.get_tracer(DEFAULT_TRACER_NAME)


def get_tracer(name: str = DEFAULT_TRACER_NAME) -> Tracer:
    """Return an OpenTelemetry Tracer backed by the configured SDK TracerProvider."""
    provider = _ensure_tracer_provider()
    return provider.get_tracer(name)


@contextmanager
def start_as_current_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    *,
    tracer_name: str = DEFAULT_TRACER_NAME,
    kind: SpanKind = SpanKind.INTERNAL,
) -> Iterator[Span]:
    """Start and activate a child OpenTelemetry span with PII-sanitized attributes."""
    tracer = get_tracer(tracer_name)
    sanitized_attrs = _sanitize_span_attributes(attributes) if attributes else None
    with tracer.start_as_current_span(
        name,
        kind=kind,
        attributes=sanitized_attrs,
    ) as span:
        yield span


def get_current_trace_ids() -> tuple[str, str]:
    """Return active W3C 32-hex ``trace_id`` and 16-hex ``span_id``.

    When called inside an active OpenTelemetry span, returns the span's exact
    W3C hex identifiers. When called outside an active span, returns a stable
    context-local 32-hex and 16-hex pair so correlated logs remain valid.
    """
    span = trace.get_current_span()
    span_context = span.get_span_context()
    if span_context is not None and span_context.is_valid:
        return (
            trace.format_trace_id(span_context.trace_id),
            trace.format_span_id(span_context.span_id),
        )
    fallback = _FALLBACK_TRACE_CTX.get()
    if fallback is None:
        fallback = (uuid.uuid4().hex, uuid.uuid4().hex[:16])
        _FALLBACK_TRACE_CTX.set(fallback)
    return fallback


def _serialize_readable_span(span: ReadableSpan) -> dict[str, Any]:
    """Serialize a finished ReadableSpan into a PII-scrubbed JSON-ready dictionary."""
    ctx = span.get_span_context()
    trace_id = (
        trace.format_trace_id(ctx.trace_id)
        if ctx is not None and ctx.is_valid
        else "0" * 32
    )
    span_id = (
        trace.format_span_id(ctx.span_id)
        if ctx is not None and ctx.is_valid
        else "0" * 16
    )
    parent = span.parent
    parent_span_id = (
        trace.format_span_id(parent.span_id)
        if parent is not None and parent.is_valid
        else None
    )
    raw_attrs = dict(span.attributes or {})
    clean_attrs: dict[str, Any] = {}
    for key, val in raw_attrs.items():
        str_key = str(key)
        if str_key.lower() in _SENSITIVE_ATTR_KEYS and isinstance(val, str) and val.strip():
            clean_attrs[str_key] = "[REDACTED_SECRET]"
        elif isinstance(val, str):
            clean_attrs[str_key] = _scrub_string(val)
        elif isinstance(val, (bool, int, float)) or val is None:
            clean_attrs[str_key] = val
        elif isinstance(val, Sequence) and not isinstance(val, (bytes, bytearray)):
            clean_attrs[str_key] = [
                _scrub_string(item) if isinstance(item, str) else item
                for item in val
            ]
        else:
            clean_attrs[str_key] = _scrub_string(str(val))
    duration_ms = (
        round((span.end_time - span.start_time) / 1_000_000.0, 3)
        if span.start_time is not None and span.end_time is not None
        else 0.0
    )
    return {
        "name": _scrub_string(span.name),
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "duration_ms": duration_ms,
        "status": span.status.status_code.name,
        "attributes": clean_attrs,
    }


def get_recorded_spans(limit: int = 200) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent finished spans as JSON-ready dicts."""
    if limit <= 0:
        return []
    _ensure_tracer_provider()
    if _IN_MEMORY_EXPORTER is None:
        return []
    finished = _IN_MEMORY_EXPORTER.get_finished_spans()
    return [_serialize_readable_span(span) for span in finished[-limit:]]


def clear_recorded_spans() -> None:
    """Clear all spans stored in the in-memory span exporter."""
    if _IN_MEMORY_EXPORTER is not None:
        _IN_MEMORY_EXPORTER.clear()
