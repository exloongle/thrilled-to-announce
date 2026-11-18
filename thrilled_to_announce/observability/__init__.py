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
"""Observability package: PII scrubbing, structured JSON logging, and OTel tracing."""

from thrilled_to_announce.observability.logging import (
    clear_telemetry_logs,
    configure_logging,
    get_logger,
    get_telemetry_logs,
)
from thrilled_to_announce.observability.pii_scrubber import (
    REDACTED_EMAIL,
    REDACTED_PHONE,
    REDACTED_SALARY,
    REDACTED_SECRET,
    REDACTED_SSN,
    PiiScrubber,
    ScrubResult,
    get_default_scrubber,
    redact_pii_data,
    reset_default_scrubber,
    scrub_text,
    structlog_pii_processor,
)
from thrilled_to_announce.observability.tracing import (
    clear_recorded_spans,
    configure_tracing,
    get_current_trace_ids,
    get_recorded_spans,
    get_tracer,
    start_as_current_span,
)

__all__ = [
    "PiiScrubber",
    "REDACTED_EMAIL",
    "REDACTED_PHONE",
    "REDACTED_SALARY",
    "REDACTED_SECRET",
    "REDACTED_SSN",
    "ScrubResult",
    "clear_recorded_spans",
    "clear_telemetry_logs",
    "configure_logging",
    "configure_tracing",
    "get_current_trace_ids",
    "get_default_scrubber",
    "get_logger",
    "get_recorded_spans",
    "get_telemetry_logs",
    "get_tracer",
    "redact_pii_data",
    "reset_default_scrubber",
    "scrub_text",
    "start_as_current_span",
    "structlog_pii_processor",
]
