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
"""Active PII redaction combining Google Cloud Sensitive Data Protection (DLP) and deterministic regexes."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import re
from typing import Any

from google.cloud import dlp_v2
from pydantic import BaseModel, ConfigDict, Field

from thrilled_to_announce.config.settings import get_settings

REDACTED_EMAIL = "[REDACTED_EMAIL]"
REDACTED_PHONE = "[REDACTED_PHONE]"
REDACTED_SSN = "[REDACTED_SSN]"
REDACTED_SALARY = "[REDACTED_SALARY]"
REDACTED_SECRET = "[REDACTED_SECRET]"

_MARKER_TO_CATEGORY: dict[str, str] = {
    REDACTED_SECRET: "SECRET",
    REDACTED_EMAIL: "EMAIL",
    REDACTED_SSN: "SSN",
    REDACTED_PHONE: "PHONE",
    REDACTED_SALARY: "SALARY",
}

_DLP_INFO_TYPE_TO_MARKER: dict[str, tuple[str, str]] = {
    "EMAIL_ADDRESS": ("EMAIL", REDACTED_EMAIL),
    "PHONE_NUMBER": ("PHONE", REDACTED_PHONE),
    "US_SOCIAL_SECURITY_NUMBER": ("SSN", REDACTED_SSN),
    "AUTH_TOKEN": ("SECRET", REDACTED_SECRET),
    "GCP_API_KEY": ("SECRET", REDACTED_SECRET),
    "GENERIC_API_KEY": ("SECRET", REDACTED_SECRET),
}

SENSITIVE_DICT_KEYS = frozenset({
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

_SECRET_PATTERN = re.compile(
    r"(?:"
    r"-----BEGIN [A-Z0-9 ]+-----[\s\S]*?-----END [A-Z0-9 ]+-----"
    r"|\bAIza[0-9A-Za-z\-_]{20,}\b"
    r"|\bsk-(?:ant-|proj-)?[A-Za-z0-9_\-]{16,}\b"
    r"|\bgh[pousr]_[A-Za-z0-9]{20,}\b"
    r"|\bxox[abpr]-[A-Za-z0-9\-]{10,}\b"
    r"|\bBearer\s+(?!\[REDACTED_)[A-Za-z0-9\-._~+/]{16,}=*"
    r"|(?i:\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)"
    r"\s*[:=]\s*['\"]?(?!\[REDACTED_)[A-Za-z0-9\-._~+/]{12,}['\"]?)"
    r")"
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,24}\b"
)

_SSN_PATTERN = re.compile(
    r"(?<!\d)(?!000|666|9\d\d)\d{3}[-\s](?!00)\d{2}[-\s](?!0000)\d{4}(?!\d)"
)

_PHONE_PATTERN = re.compile(
    r"(?<!\w)(?:\+\d{1,3}[\s.\-]?)?(?:\(\d{3}\)[\s.\-]?|\d{3}[\s.\-])\d{3}[\s.\-]\d{4}(?!\d)"
)

_CORP_METRIC_BEFORE_RE = re.compile(
    r"\b(?:revenue|arr|mrr|valuation|funding|raised|seed|series\s+[a-f]|"
    r"budget|ebitda|profit|bookings?|pipeline|market\s+cap)"
    r"(?:\s+of|\s+is|\s+at)?[\s:(\-]*$",
    re.IGNORECASE,
)

_NUM_END_GUARD = (
    r"(?![,\d])"
    r"(?!\s*(?:[MBT]\b|MM\b|bn\b|mn\b|million\b|billion\b|trillion\b|"
    r"ARR\b|MRR\b|valuation\b|revenue\b|funding\b|round\b|series\b|budget\b|bookings?\b))"
)

_SALARY_PATTERN = re.compile(
    r"(?:"
    r"(?P<prefix>\b(?:salary|pay|comp|compensation|base|wage|offer|package|making|earning|paid)"
    r"\s+(?:of\s+|is\s+|at\s+|around\s+|was\s+)?)"
    r"\$\d{1,7}(?:,\d{3})*(?:\.\d{1,2})?[kK]?\b"
    + _NUM_END_GUARD
    + r"|\$(?:\d{2,3},\d{3}|[1-9],\d{3},\d{3})(?:\.\d{1,2})?"
    + _NUM_END_GUARD
    + r"|\$(?:\d{1},\d{3}(?:\.\d{1,2})?|\d{2,3}(?:\.\d+)?[kK]\b)(?![,\d])"
    r"(?=\s*(?:base\b|salary\b|pay\b|comp\b|compensation\b|tc\b|total\s+comp\b|"
    r"bonus\b|offer\b|stipend\b|per\s+(?:year|month|week)\b|"
    r"/\s*(?:yr|year|mo|month|wk)\b|a\s+(?:year|month)\b|annually\b|monthly\b))"
    r"|\$\d{5,6}\.\d{2}\b"
    + _NUM_END_GUARD
    + r"|\$\d{5,6}\b(?![,\d])(?=\s*(?:base\b|salary\b|pay\b|comp\b|compensation\b|tc\b|"
    r"per\s+year\b|/\s*yr\b|/\s*year\b|annually\b))"
    r"|\$\d{2,3}(?:\.\d{1,2})?\s*(?:/\s*hr\b|/\s*hour\b|per\s+hour\b|an\s+hour\b)"
    r")",
    re.IGNORECASE,
)


class ScrubResult(BaseModel):
    """Structured result of scrubbing PII and credentials from text."""

    model_config = ConfigDict(extra="forbid")

    scrubbed_text: str
    redactions_count: int = Field(default=0, ge=0)
    redacted_types: list[str] = Field(default_factory=list)


class PiiScrubber:
    """Scrubs emails, phones, SSNs, compensation figures, and secrets via Cloud DLP and regexes."""

    def __init__(
        self,
        *,
        enable_cloud_dlp: bool | None = None,
        project_id: str | None = None,
        location: str = "global",
        dlp_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.enable_cloud_dlp = (
            settings.enable_cloud_dlp if enable_cloud_dlp is None else enable_cloud_dlp
        )
        self.project_id = (
            project_id.strip()
            if project_id
            and project_id.strip()
            and project_id.strip().lower() != "your_project_id"
            else settings.google_cloud_project
        )
        self.location = location or "global"
        self._dlp_client = dlp_client

    def _get_dlp_client(self) -> Any:
        if self._dlp_client is None:
            self._dlp_client = dlp_v2.DlpServiceClient()
        return self._dlp_client

    @staticmethod
    def _build_dlp_configs() -> tuple[dlp_v2.InspectConfig, dlp_v2.DeidentifyConfig]:
        inspect_config = dlp_v2.InspectConfig(
            info_types=[
                dlp_v2.InfoType(name=name) for name in _DLP_INFO_TYPE_TO_MARKER
            ],
            min_likelihood=dlp_v2.Likelihood.POSSIBLE,
        )
        transformations = [
            dlp_v2.InfoTypeTransformations.InfoTypeTransformation(
                info_types=[dlp_v2.InfoType(name=info_type_name)],
                primitive_transformation=dlp_v2.PrimitiveTransformation(
                    replace_config=dlp_v2.ReplaceValueConfig(
                        new_value=dlp_v2.Value(string_value=marker)
                    )
                ),
            )
            for info_type_name, (_, marker) in _DLP_INFO_TYPE_TO_MARKER.items()
        ]
        deidentify_config = dlp_v2.DeidentifyConfig(
            info_type_transformations=dlp_v2.InfoTypeTransformations(
                transformations=transformations
            )
        )
        return inspect_config, deidentify_config

    def _scrub_with_cloud_dlp(self, text: str) -> tuple[str, int, list[str]]:
        if not self.enable_cloud_dlp or not text:
            return text, 0, []
        effective_project = self.project_id
        if not effective_project and self._dlp_client is None:
            return text, 0, []

        parent = dlp_v2.DlpServiceClient.common_location_path(
            effective_project or "local-test-project", self.location
        )
        inspect_config, deidentify_config = self._build_dlp_configs()
        try:
            client = self._get_dlp_client()
            response = client.deidentify_content(
                request={
                    "parent": parent,
                    "inspect_config": inspect_config,
                    "deidentify_config": deidentify_config,
                    "item": {"value": text},
                }
            )
            deidentified = getattr(getattr(response, "item", None), "value", None)
            if not isinstance(deidentified, str):
                return text, 0, []

            count = 0
            types: list[str] = []
            overview = getattr(response, "overview", None)
            summaries = getattr(overview, "transformation_summaries", None) or []
            for summary in summaries:
                info_name = getattr(getattr(summary, "info_type", None), "name", "")
                category = _DLP_INFO_TYPE_TO_MARKER.get(info_name, ("SECRET", ""))[0]
                summary_count = sum(
                    int(getattr(res, "count", 0) or 0)
                    for res in (getattr(summary, "results", None) or [])
                )
                if summary_count > 0:
                    count += summary_count
                    if category not in types:
                        types.append(category)

            if count == 0 and deidentified != text:
                for marker, category in _MARKER_TO_CATEGORY.items():
                    delta = deidentified.count(marker) - text.count(marker)
                    if delta > 0:
                        count += delta
                        if category not in types:
                            types.append(category)
                if count == 0:
                    count = 1
                    types.append("CLOUD_DLP")

            return deidentified, count, types
        except Exception:
            return text, 0, []

    @staticmethod
    def _scrub_with_regex(text: str) -> tuple[str, int, list[str]]:
        if not text:
            return text, 0, []

        current = text
        total_count = 0
        found_types: list[str] = []

        for category, marker, pattern in (
            ("SECRET", REDACTED_SECRET, _SECRET_PATTERN),
            ("EMAIL", REDACTED_EMAIL, _EMAIL_PATTERN),
            ("SSN", REDACTED_SSN, _SSN_PATTERN),
            ("PHONE", REDACTED_PHONE, _PHONE_PATTERN),
        ):
            current, replacements = pattern.subn(marker, current)
            if replacements > 0:
                total_count += replacements
                if category not in found_types:
                    found_types.append(category)

        salary_hits = 0

        def _replace_salary(match: re.Match[str]) -> str:
            nonlocal salary_hits
            prefix = match.group("prefix") or ""
            if not prefix:
                before_window = match.string[max(0, match.start() - 40) : match.start()]
                if _CORP_METRIC_BEFORE_RE.search(before_window):
                    return match.group(0)
            salary_hits += 1
            return f"{prefix}{REDACTED_SALARY}"

        current = _SALARY_PATTERN.sub(_replace_salary, current)
        if salary_hits > 0:
            total_count += salary_hits
            if "SALARY" not in found_types:
                found_types.append("SALARY")

        return current, total_count, found_types

    def scrub(self, text: str) -> ScrubResult:
        """Scrub PII and secrets from a string using Cloud DLP (when enabled) and local regexes."""
        if not text:
            return ScrubResult(
                scrubbed_text=text, redactions_count=0, redacted_types=[]
            )

        working_text, dlp_count, dlp_types = self._scrub_with_cloud_dlp(text)
        final_text, regex_count, regex_types = self._scrub_with_regex(working_text)

        merged_types: list[str] = []
        for item in (*dlp_types, *regex_types):
            if item not in merged_types:
                merged_types.append(item)

        return ScrubResult(
            scrubbed_text=final_text,
            redactions_count=dlp_count + regex_count,
            redacted_types=merged_types,
        )

    def scrub_payload(self, value: Any) -> tuple[Any, int, list[str]]:
        """Recursively scrub strings, mappings, and sequences, returning ``(scrubbed, count, types)``."""
        if isinstance(value, str):
            res = self.scrub(value)
            return res.scrubbed_text, res.redactions_count, list(res.redacted_types)

        if isinstance(value, BaseModel):
            return self.scrub_payload(value.model_dump())

        if isinstance(value, Mapping):
            scrubbed_dict: dict[Any, Any] = {}
            total = 0
            types: list[str] = []
            for k, v in value.items():
                key_lower = k.lower() if isinstance(k, str) else ""
                if (
                    key_lower in SENSITIVE_DICT_KEYS
                    and isinstance(v, str)
                    and v.strip()
                    and v.strip() != REDACTED_SECRET
                ):
                    scrubbed_dict[k] = REDACTED_SECRET
                    total += 1
                    if "SECRET" not in types:
                        types.append("SECRET")
                else:
                    sub_val, sub_cnt, sub_types = self.scrub_payload(v)
                    scrubbed_dict[k] = sub_val
                    total += sub_cnt
                    for t in sub_types:
                        if t not in types:
                            types.append(t)
            return scrubbed_dict, total, types

        if isinstance(value, list):
            out_list: list[Any] = []
            total = 0
            types = []
            for elem in value:
                sub_val, sub_cnt, sub_types = self.scrub_payload(elem)
                out_list.append(sub_val)
                total += sub_cnt
                for t in sub_types:
                    if t not in types:
                        types.append(t)
            return out_list, total, types

        if isinstance(value, tuple):
            sub_list, total, types = self.scrub_payload(list(value))
            return tuple(sub_list), total, types

        return value, 0, []

    def scrub_value(self, value: Any) -> Any:
        """Recursively scrub a nested structure and return only the sanitized structure."""
        scrubbed, _, _ = self.scrub_payload(value)
        return scrubbed


_DEFAULT_SCRUBBERS: dict[tuple[bool, str | None], PiiScrubber] = {}


def reset_default_scrubber() -> None:
    """Clear cached ``PiiScrubber`` instances."""
    _DEFAULT_SCRUBBERS.clear()


def get_default_scrubber() -> PiiScrubber:
    """Return a ``PiiScrubber`` aligned with current ``AppSettings``."""
    settings = get_settings()
    key = (settings.enable_cloud_dlp, settings.google_cloud_project)
    scrubber = _DEFAULT_SCRUBBERS.get(key)
    if scrubber is None:
        scrubber = PiiScrubber(
            enable_cloud_dlp=settings.enable_cloud_dlp,
            project_id=settings.google_cloud_project,
        )
        _DEFAULT_SCRUBBERS[key] = scrubber
    return scrubber


def scrub_text(text: str) -> ScrubResult:
    """Scrub PII and secrets from ``text`` using the default ``PiiScrubber``."""
    return get_default_scrubber().scrub(text)


def redact_pii_data(value: Any) -> Any:
    """Convenience recursive redaction helper for loggers, ADK callbacks, and memory stores."""
    return get_default_scrubber().scrub_value(value)


def structlog_pii_processor(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """``structlog`` processor that scrubs PII across all event fields in-place."""
    del logger, method_name
    scrubber = get_default_scrubber()
    scrubbed_map, _, _ = scrubber.scrub_payload(dict(event_dict))
    event_dict.clear()
    event_dict.update(scrubbed_map)
    return event_dict
