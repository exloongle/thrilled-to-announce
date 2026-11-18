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
"""ADK telemetry plugin capturing paired INTENT and OUTCOME logs with OTel spans and error recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
import json
import time
from types import TracebackType
from typing import Any, Literal
import uuid

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, set_span_in_context

from thrilled_to_announce.observability.logging import get_logger
from thrilled_to_announce.observability.pii_scrubber import (
    redact_pii_data,
    scrub_text,
)
from thrilled_to_announce.observability.tracing import (
    get_current_trace_ids,
    get_tracer,
)

_MAX_SUMMARY_CHARS = 500


def _summarize_payload(payload: Any) -> str:
    """Serialize and PII-scrub an argument or result payload to a compact string."""
    if payload is None:
        return ""
    scrubbed_obj = redact_pii_data(payload)
    try:
        rendered = (
            scrubbed_obj
            if isinstance(scrubbed_obj, str)
            else json.dumps(scrubbed_obj, default=str, sort_keys=True)
        )
    except Exception:
        rendered = str(scrubbed_obj)
    cleaned = scrub_text(rendered).scrubbed_text
    if len(cleaned) > _MAX_SUMMARY_CHARS:
        return f"{cleaned[:_MAX_SUMMARY_CHARS]}..."
    return cleaned


@dataclass
class _InflightOperation:
    """Tracks an active agent or tool span so INTENT and OUTCOME share exact IDs."""

    correlation_id: str
    trace_id: str
    span_id: str
    start_ns: int
    span: Any
    otel_ctx: Any
    session_id: str
    user_id: str
    subject_name: str
    operation_kind: Literal["agent", "tool"]


class IntentOutcomeContext:
    """Dual sync/async context manager emitting paired INTENT and OUTCOME telemetry."""

    def __init__(
        self,
        operation_name: str,
        *,
        operation_kind: Literal["tool", "agent", "pipeline", "http"] = "tool",
        session_id: str = "default_session",
        user_id: str = "default_user",
        args: Mapping[str, Any] | None = None,
        correlation_id: str | None = None,
    ) -> None:
        self.operation_name = operation_name
        self.operation_kind = operation_kind
        self.session_id = session_id
        self.user_id = user_id
        self._args = dict(args) if args else {}
        prefix = operation_kind[:3]
        self.correlation_id = correlation_id or f"{prefix}_{uuid.uuid4().hex[:12]}"
        self.trace_id = "0" * 32
        self.span_id = "0" * 16
        self._span_cm: Any = None
        self._span: Any = None
        self._start_ns = 0
        self._outcome_status = "ok"
        self._outcome_result: Any = None
        self._outcome_error: str | None = None

    def set_outcome(
        self,
        *,
        status: str = "ok",
        result: Any = None,
        error_message: str | None = None,
    ) -> None:
        """Record explicit outcome metadata prior to context exit."""
        self._outcome_status = status
        if result is not None:
            self._outcome_result = result
        if error_message is not None:
            self._outcome_error = scrub_text(error_message).scrubbed_text

    def _enter(self) -> IntentOutcomeContext:
        self._start_ns = time.perf_counter_ns()
        tracer = get_tracer("thrilled_to_announce")
        args_summary = _summarize_payload(self._args)
        span_attrs: dict[str, Any] = {
            "thrilled.operation_kind": self.operation_kind,
            "thrilled.operation_name": self.operation_name,
            "thrilled.correlation_id": self.correlation_id,
            "thrilled.session_id": self.session_id,
            "thrilled.user_id": self.user_id,
        }
        if args_summary:
            span_attrs["thrilled.args_summary"] = args_summary

        self._span_cm = tracer.start_as_current_span(
            f"{self.operation_kind}.{self.operation_name}",
            attributes=span_attrs,
        )
        self._span = self._span_cm.__enter__()
        self.trace_id, self.span_id = get_current_trace_ids()

        extra_fields: dict[str, Any] = {}
        if self.operation_kind == "tool":
            extra_fields["tool_name"] = self.operation_name
        elif self.operation_kind == "agent":
            extra_fields["agent_name"] = self.operation_name

        get_logger(__name__).info(
            f"{self.operation_kind}_intent_planned",
            phase="INTENT",
            correlation_id=self.correlation_id,
            trace_id=self.trace_id,
            span_id=self.span_id,
            session_id=self.session_id,
            user_id=self.user_id,
            operation_name=self.operation_name,
            args_summary=args_summary,
            **extra_fields,
        )
        return self

    def _exit(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        latency_ms = round((time.perf_counter_ns() - self._start_ns) / 1_000_000.0, 3)
        extra_fields: dict[str, Any] = {}
        if self.operation_kind == "tool":
            extra_fields["tool_name"] = self.operation_name
        elif self.operation_kind == "agent":
            extra_fields["agent_name"] = self.operation_name

        if exc_val is not None:
            scrubbed_err = scrub_text(str(exc_val)).scrubbed_text
            if self._span is not None:
                self._span.set_status(Status(StatusCode.ERROR, scrubbed_err))
            get_logger(__name__).info(
                f"{self.operation_kind}_outcome_completed",
                phase="OUTCOME",
                status="error",
                correlation_id=self.correlation_id,
                trace_id=self.trace_id,
                span_id=self.span_id,
                session_id=self.session_id,
                user_id=self.user_id,
                operation_name=self.operation_name,
                latency_ms=latency_ms,
                error_type=exc_type.__name__ if exc_type is not None else "Exception",
                error_message=scrubbed_err,
                **extra_fields,
            )
        else:
            if self._span is not None:
                if self._outcome_status in {"error", "recoverable_error"}:
                    self._span.set_status(
                        Status(StatusCode.ERROR, self._outcome_error or self._outcome_status)
                    )
                else:
                    self._span.set_status(Status(StatusCode.OK))
            get_logger(__name__).info(
                f"{self.operation_kind}_outcome_completed",
                phase="OUTCOME",
                status=self._outcome_status,
                correlation_id=self.correlation_id,
                trace_id=self.trace_id,
                span_id=self.span_id,
                session_id=self.session_id,
                user_id=self.user_id,
                operation_name=self.operation_name,
                latency_ms=latency_ms,
                result_summary=_summarize_payload(self._outcome_result),
                error_message=self._outcome_error,
                **extra_fields,
            )

        if self._span_cm is not None:
            self._span_cm.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __enter__(self) -> IntentOutcomeContext:
        return self._enter()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        return self._exit(exc_type, exc_val, exc_tb)

    async def __aenter__(self) -> IntentOutcomeContext:
        return self._enter()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        return self._exit(exc_type, exc_val, exc_tb)


def record_intent_and_outcome(
    operation_name: str,
    *,
    operation_kind: Literal["tool", "agent", "pipeline", "http"] = "tool",
    session_id: str = "default_session",
    user_id: str = "default_user",
    args: Mapping[str, Any] | None = None,
    correlation_id: str | None = None,
) -> IntentOutcomeContext:
    """Create a sync/async context manager that logs paired INTENT and OUTCOME records."""
    return IntentOutcomeContext(
        operation_name,
        operation_kind=operation_kind,
        session_id=session_id,
        user_id=user_id,
        args=args,
        correlation_id=correlation_id,
    )


class IntentOutcomeTelemetryPlugin(BasePlugin):
    """Emits paired INTENT and OUTCOME structured JSON logs and OTel spans across ADK turns."""

    def __init__(self, name: str = "intent_outcome_telemetry_plugin") -> None:
        super().__init__(name=name)
        self._inflight_agents: dict[tuple[str, str], _InflightOperation] = {}
        self._invocation_agent_stack: dict[str, list[tuple[str, str]]] = {}
        self._inflight_tools: dict[tuple[str, str], _InflightOperation] = {}
        self._handled_tool_errors: set[tuple[str, str]] = set()

    @staticmethod
    def _format_span_ids(span: Any) -> tuple[str, str]:
        sc = span.get_span_context()
        if sc is not None and sc.is_valid:
            return trace.format_trace_id(sc.trace_id), trace.format_span_id(sc.span_id)
        return get_current_trace_ids()

    def _parent_context_for_agent(self, invocation_id: str) -> Any:
        stack = self._invocation_agent_stack.get(invocation_id)
        if stack:
            parent_op = self._inflight_agents.get(stack[-1])
            if parent_op is not None:
                return parent_op.otel_ctx
        return otel_context.get_current()

    def _parent_context_for_tool(self, invocation_id: str, agent_name: str) -> Any:
        direct_op = self._inflight_agents.get((invocation_id, agent_name))
        if direct_op is not None:
            return direct_op.otel_ctx
        return self._parent_context_for_agent(invocation_id)

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Start an OpenTelemetry agent span and emit a structured INTENT log entry."""
        invocation_id = callback_context.invocation_id
        agent_key = (invocation_id, agent.name)
        correlation_id = f"agt_{uuid.uuid4().hex[:12]}"
        parent_ctx = self._parent_context_for_agent(invocation_id)

        span = get_tracer("thrilled_to_announce").start_span(
            f"agent.{agent.name}",
            context=parent_ctx,
            attributes={
                "thrilled.operation_kind": "agent",
                "thrilled.agent_name": agent.name,
                "thrilled.correlation_id": correlation_id,
                "thrilled.session_id": callback_context.session.id,
                "thrilled.user_id": callback_context.user_id,
                "thrilled.invocation_id": invocation_id,
            },
        )
        trace_id, span_id = self._format_span_ids(span)
        op = _InflightOperation(
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            start_ns=time.perf_counter_ns(),
            span=span,
            otel_ctx=set_span_in_context(span, parent_ctx),
            session_id=callback_context.session.id,
            user_id=callback_context.user_id,
            subject_name=agent.name,
            operation_kind="agent",
        )
        self._inflight_agents[agent_key] = op
        self._invocation_agent_stack.setdefault(invocation_id, []).append(agent_key)

        with trace.use_span(span, end_on_exit=False):
            get_logger(__name__).info(
                "agent_intent_planned",
                phase="INTENT",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=span_id,
                session_id=callback_context.session.id,
                user_id=callback_context.user_id,
                agent_name=agent.name,
                invocation_id=invocation_id,
            )
        return None

    async def after_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Close the active OpenTelemetry agent span and emit a paired OUTCOME log entry."""
        invocation_id = callback_context.invocation_id
        agent_key = (invocation_id, agent.name)
        op = self._inflight_agents.pop(agent_key, None)
        stack = self._invocation_agent_stack.get(invocation_id)
        if stack and agent_key in stack:
            stack.remove(agent_key)

        if op is None:
            trace_id, span_id = get_current_trace_ids()
            get_logger(__name__).info(
                "agent_outcome_completed",
                phase="OUTCOME",
                status="ok",
                correlation_id=f"agt_{uuid.uuid4().hex[:12]}",
                trace_id=trace_id,
                span_id=span_id,
                session_id=callback_context.session.id,
                user_id=callback_context.user_id,
                agent_name=agent.name,
                latency_ms=0.0,
            )
            return None

        latency_ms = round((time.perf_counter_ns() - op.start_ns) / 1_000_000.0, 3)
        op.span.set_status(Status(StatusCode.OK))
        with trace.use_span(op.span, end_on_exit=False):
            get_logger(__name__).info(
                "agent_outcome_completed",
                phase="OUTCOME",
                status="ok",
                correlation_id=op.correlation_id,
                trace_id=op.trace_id,
                span_id=op.span_id,
                session_id=callback_context.session.id,
                user_id=callback_context.user_id,
                agent_name=agent.name,
                invocation_id=invocation_id,
                latency_ms=latency_ms,
            )
        op.span.end()
        return None

    async def on_agent_error_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
        error: Exception,
    ) -> None:
        """Close any in-flight agent span with ERROR status on unhandled agent error."""
        invocation_id = callback_context.invocation_id
        agent_key = (invocation_id, agent.name)
        op = self._inflight_agents.pop(agent_key, None)
        stack = self._invocation_agent_stack.get(invocation_id)
        if stack and agent_key in stack:
            stack.remove(agent_key)
        if op is None:
            return
        scrubbed_err = scrub_text(str(error)).scrubbed_text
        latency_ms = round((time.perf_counter_ns() - op.start_ns) / 1_000_000.0, 3)
        op.span.set_status(Status(StatusCode.ERROR, scrubbed_err))
        with trace.use_span(op.span, end_on_exit=False):
            get_logger(__name__).info(
                "agent_outcome_completed",
                phase="OUTCOME",
                status="error",
                correlation_id=op.correlation_id,
                trace_id=op.trace_id,
                span_id=op.span_id,
                session_id=callback_context.session.id,
                user_id=callback_context.user_id,
                agent_name=agent.name,
                latency_ms=latency_ms,
                error_type=type(error).__name__,
                error_message=scrubbed_err,
            )
        op.span.end()

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Start a child OpenTelemetry tool span and emit a structured INTENT log entry."""
        invocation_id = tool_context.invocation_id
        call_key = (invocation_id, tool_context.function_call_id or tool.name)
        self._handled_tool_errors.discard(call_key)

        correlation_id = f"tol_{uuid.uuid4().hex[:12]}"
        parent_ctx = self._parent_context_for_tool(
            invocation_id, tool_context.agent_name
        )
        args_summary = _summarize_payload(tool_args)

        span = get_tracer("thrilled_to_announce").start_span(
            f"tool.{tool.name}",
            context=parent_ctx,
            attributes={
                "thrilled.operation_kind": "tool",
                "thrilled.tool_name": tool.name,
                "thrilled.agent_name": tool_context.agent_name,
                "thrilled.correlation_id": correlation_id,
                "thrilled.session_id": tool_context.session.id,
                "thrilled.user_id": tool_context.user_id,
                "thrilled.args_summary": args_summary,
            },
        )
        trace_id, span_id = self._format_span_ids(span)
        op = _InflightOperation(
            correlation_id=correlation_id,
            trace_id=trace_id,
            span_id=span_id,
            start_ns=time.perf_counter_ns(),
            span=span,
            otel_ctx=set_span_in_context(span, parent_ctx),
            session_id=tool_context.session.id,
            user_id=tool_context.user_id,
            subject_name=tool.name,
            operation_kind="tool",
        )
        self._inflight_tools[call_key] = op

        with trace.use_span(span, end_on_exit=False):
            get_logger(__name__).info(
                "tool_intent_planned",
                phase="INTENT",
                correlation_id=correlation_id,
                trace_id=trace_id,
                span_id=span_id,
                session_id=tool_context.session.id,
                user_id=tool_context.user_id,
                agent_name=tool_context.agent_name,
                tool_name=tool.name,
                args_summary=args_summary,
            )
        return None

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Close the child OpenTelemetry tool span and emit a paired OUTCOME log entry."""
        del tool_args
        call_key = (
            tool_context.invocation_id,
            tool_context.function_call_id or tool.name,
        )
        if call_key in self._handled_tool_errors:
            self._handled_tool_errors.discard(call_key)
            return None

        op = self._inflight_tools.pop(call_key, None)
        outcome_status = (
            str(result.get("status", "ok")) if isinstance(result, Mapping) else "ok"
        )
        result_summary = _summarize_payload(result)

        if op is None:
            trace_id, span_id = get_current_trace_ids()
            get_logger(__name__).info(
                "tool_outcome_completed",
                phase="OUTCOME",
                status=outcome_status,
                correlation_id=f"tol_{uuid.uuid4().hex[:12]}",
                trace_id=trace_id,
                span_id=span_id,
                session_id=tool_context.session.id,
                user_id=tool_context.user_id,
                agent_name=tool_context.agent_name,
                tool_name=tool.name,
                latency_ms=0.0,
                result_summary=result_summary,
            )
            return None

        latency_ms = round((time.perf_counter_ns() - op.start_ns) / 1_000_000.0, 3)
        if outcome_status in {"error", "recoverable_error"}:
            op.span.set_status(Status(StatusCode.ERROR, outcome_status))
        else:
            op.span.set_status(Status(StatusCode.OK))

        with trace.use_span(op.span, end_on_exit=False):
            get_logger(__name__).info(
                "tool_outcome_completed",
                phase="OUTCOME",
                status=outcome_status,
                correlation_id=op.correlation_id,
                trace_id=op.trace_id,
                span_id=op.span_id,
                session_id=tool_context.session.id,
                user_id=tool_context.user_id,
                agent_name=tool_context.agent_name,
                tool_name=tool.name,
                latency_ms=latency_ms,
                result_summary=result_summary,
            )
        op.span.end()
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Convert unhandled tool exceptions into guided recovery payloads and emit OUTCOME."""
        del tool_args
        call_key = (
            tool_context.invocation_id,
            tool_context.function_call_id or tool.name,
        )
        op = self._inflight_tools.pop(call_key, None)
        scrubbed_error = scrub_text(str(error)).scrubbed_text
        latency_ms = (
            round((time.perf_counter_ns() - op.start_ns) / 1_000_000.0, 3)
            if op is not None
            else 0.0
        )

        if isinstance(error, asyncio.CancelledError):
            if op is not None:
                op.span.set_status(Status(StatusCode.ERROR, "cancelled"))
                with trace.use_span(op.span, end_on_exit=False):
                    get_logger(__name__).info(
                        "tool_outcome_completed",
                        phase="OUTCOME",
                        status="cancelled",
                        correlation_id=op.correlation_id,
                        trace_id=op.trace_id,
                        span_id=op.span_id,
                        session_id=tool_context.session.id,
                        user_id=tool_context.user_id,
                        agent_name=tool_context.agent_name,
                        tool_name=tool.name,
                        latency_ms=latency_ms,
                    )
                op.span.end()
            return None

        self._handled_tool_errors.add(call_key)

        if op is not None:
            op.span.set_status(Status(StatusCode.ERROR, scrubbed_error))
            with trace.use_span(op.span, end_on_exit=False):
                get_logger(__name__).info(
                    "tool_outcome_completed",
                    phase="OUTCOME",
                    status="recoverable_error",
                    correlation_id=op.correlation_id,
                    trace_id=op.trace_id,
                    span_id=op.span_id,
                    session_id=tool_context.session.id,
                    user_id=tool_context.user_id,
                    agent_name=tool_context.agent_name,
                    tool_name=tool.name,
                    latency_ms=latency_ms,
                    error_type=type(error).__name__,
                    error_message=scrubbed_error,
                )
            op.span.end()
        else:
            trace_id, span_id = get_current_trace_ids()
            get_logger(__name__).info(
                "tool_outcome_completed",
                phase="OUTCOME",
                status="recoverable_error",
                correlation_id=f"tol_{uuid.uuid4().hex[:12]}",
                trace_id=trace_id,
                span_id=span_id,
                session_id=tool_context.session.id,
                user_id=tool_context.user_id,
                agent_name=tool_context.agent_name,
                tool_name=tool.name,
                latency_ms=0.0,
                error_type=type(error).__name__,
                error_message=scrubbed_error,
            )

        return {
            "status": "recoverable_error",
            "tool_name": tool.name,
            "error_type": type(error).__name__,
            "error_message": scrubbed_error,
            "recovery_instructions": [
                f"Please check the parameters passed to '{tool.name}' against its input schema and try again.",
                "Ensure required fields, character limits, and formatting constraints are valid before retrying.",
            ],
        }

    def _drain_invocation_orphans(
        self, invocation_id: str, *, status: str
    ) -> None:
        self._invocation_agent_stack.pop(invocation_id, None)
        stale_error_keys = [
            key for key in self._handled_tool_errors if key[0] == invocation_id
        ]
        for key in stale_error_keys:
            self._handled_tool_errors.discard(key)

        for store, event_name in (
            (self._inflight_tools, "tool_outcome_completed"),
            (self._inflight_agents, "agent_outcome_completed"),
        ):
            matching = [key for key in store if key[0] == invocation_id]
            for key in matching:
                op = store.pop(key)
                latency_ms = round(
                    (time.perf_counter_ns() - op.start_ns) / 1_000_000.0, 3
                )
                op.span.set_status(Status(StatusCode.ERROR, status))
                extra: dict[str, Any] = (
                    {"tool_name": op.subject_name}
                    if op.operation_kind == "tool"
                    else {"agent_name": op.subject_name}
                )
                with trace.use_span(op.span, end_on_exit=False):
                    get_logger(__name__).info(
                        event_name,
                        phase="OUTCOME",
                        status=status,
                        correlation_id=op.correlation_id,
                        trace_id=op.trace_id,
                        span_id=op.span_id,
                        session_id=op.session_id,
                        user_id=op.user_id,
                        latency_ms=latency_ms,
                        **extra,
                    )
                op.span.end()

    async def after_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> None:
        """Clean up any unclosed spans or tracking entries when an invocation finishes."""
        self._drain_invocation_orphans(
            invocation_context.invocation_id, status="aborted"
        )

    async def on_run_error_callback(
        self,
        *,
        invocation_context: InvocationContext,
        error: Exception,
    ) -> None:
        """Close any remaining spans with ERROR status on unhandled invocation error."""
        del error
        self._drain_invocation_orphans(
            invocation_context.invocation_id, status="error"
        )
