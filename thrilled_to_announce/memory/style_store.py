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
"""Persistent SQLite and dense vector style memory store with async consolidation.

Provides ``DenseVectorSearchIndex`` (L2-normalized ``float32`` embeddings with
cloud ``google.genai`` ``embed_content`` support and deterministic offline
n-gram feature hashing, compatible with ``pgvector`` and ``AlloyDB`` migration
paths) and ``VectorStyleMemoryStore`` backed by ``aiosqlite`` and ``sqlite3``
read-through hydration, active PII scrubbing before storage, natural-language
wording tweak consolidation, and non-blocking ``asyncio.create_task`` writes.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import struct
import sys
import time
from typing import Any, Literal
import uuid

import aiosqlite
from google import genai
from google.genai import types
import structlog

from thrilled_to_announce.schemas.contracts import (
    IdiomExample,
    PreferredFormat,
    WordingPreferenceProfile,
)

EMBEDDING_DIM: int = 128

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "i",
        "in",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "our",
        "that",
        "the",
        "this",
        "to",
        "was",
        "we",
        "with",
    }
)

_LOCAL_PII_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "SECRET",
        "[REDACTED_SECRET]",
        re.compile(
            r"(?:"
            r"AIza[0-9A-Za-z\-_]{20,}"
            r"|\bsk-(?:proj-|ant-)?[A-Za-z0-9_\-]{16,}"
            r"|\bgh[pousr]_[A-Za-z0-9]{20,}"
            r"|\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer[_-]?token)\s*[:=]\s*[^\s,;\"']+"
            r"|\bBearer\s+[A-Za-z0-9\-._~+/]{16,}=*"
            r")"
        ),
    ),
    (
        "SSN",
        "[REDACTED_SSN]",
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ),
    (
        "EMAIL",
        "[REDACTED_EMAIL]",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ),
    (
        "PHONE",
        "[REDACTED_PHONE]",
        re.compile(
            r"(?<!\w)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
        ),
    ),
    (
        "SALARY",
        "[REDACTED_SALARY]",
        re.compile(r"\$\d{1,3}(?:,\d{3})+(?:\.\d{2})?\b|\$\d{2,3}\s*[kK]\b"),
    ),
)

META_CATEGORY_TOKENS: frozenset[str] = frozenset(
    {
        "emoji",
        "emojis",
        "emoticon",
        "emoticons",
        "hashtag",
        "hashtags",
        "bullet",
        "bullets",
        "bullet point",
        "bullet points",
        "word",
        "words",
        "the word",
        "the words",
        "stiff words",
        "stiff word",
        "buzzword",
        "buzzwords",
        "phrase",
        "phrases",
        "term",
        "terms",
        "jargon",
        "corporate jargon",
        "corporate speak",
        "fluff",
        "cliche",
        "cliches",
        "cliché",
        "clichés",
        "slang",
        "profanity",
        "it",
        "this",
        "that",
        "them",
        "posts",
        "post",
        "my posts",
        "drafts",
        "draft",
        "paragraphs",
        "paragraph",
        "like",
        "such as",
        "words like",
        "terms like",
        "phrases like",
    }
)

_UNBAN_VERB_HEAD_RE = re.compile(
    r"\b(?:allow|permit|unban|you\s+can\s+use|okay\s+to\s+use)\b",
    re.IGNORECASE,
)
_UNBAN_FROM_BANNED_RE = re.compile(
    r"\b(?:remove|drop|delete)\b([^.;—!]+?)\bfrom\s+(?:(?:my|the|your|our)\s+)?"
    r"(?:banned|forbidden|blocked|disallowed|avoided)(?:\s+(?:words?|terms?|phrases?|list))?\b",
    re.IGNORECASE,
)
_BAN_VERB_HEAD_RE = re.compile(
    r"\b(?:never\s+use|do\s+not\s+use|don'?t\s+use|avoid(?:\s+using)?|stop\s+using|omit|exclude|ban)\b",
    re.IGNORECASE,
)
_REMOVE_QUOTED_BAN_RE = re.compile(
    r"\b(?:remove|drop|delete|cut|strip)\s+([^.;—!]+)",
    re.IGNORECASE,
)
_FROM_BANNED_LIST_GUARD_RE = re.compile(
    r"\bfrom\s+(?:(?:my|the|your|our)\s+)?(?:banned|forbidden|blocked|disallowed|avoided)\b",
    re.IGNORECASE,
)
_CLAUSE_DIRECTIVE_BOUNDARY_RE = re.compile(
    r"\b(?:make|keep|sound|stay|be|use|never\s+use|do\s+not\s+use|don'?t\s+use"
    r"|stop\s+using|avoid|switch|change|set|format|write|allow|permit|unban"
    r"|you\s+can\s+use|okay\s+to\s+use|enable|save|remember|store|try|turn)\b",
    re.IGNORECASE,
)
_QUOTED_ITEM_RE = re.compile(r"(?<![A-Za-z0-9])['\"]([^'\"]{1,50})['\"](?![A-Za-z0-9])")
_TRAILING_CLAUSE_CONTEXT_RE = re.compile(
    r"(?:\s+|,\s*)(?:(?:on|in|for)\s+(?:(?:my|our|all|the)\s+)?(?:linkedin(?:\s+feed|\s+posts?|\s+drafts?)?|posts?|drafts?|messages?|announcements?)"
    r"|from\s+now\s+on|going\s+forward|for\s+(?:next\s+time|future\s+(?:posts?|drafts?|conversions?))"
    r"|from\s+(?:the|my|our|your)\s+\w+(?:\s+\w+)?"
    r"|please|again|anymore|ever|at\s+all|either|now|today"
    r"|(?:and\s+)?(?:save|remember|store|keep)\s+(?:this|that)\b.*)+$",
    re.IGNORECASE,
)
_LEADING_BAN_VERB_RE = re.compile(
    r"^(?:never\s+use|do\s+not\s+use|don'?t\s+use|avoid(?:\s+using)?|stop\s+using|omit|exclude|ban)\b\s*",
    re.IGNORECASE,
)
_STANDALONE_DIRECTIVE_OR_MODIFIER_RE = re.compile(
    r"^(?:please|again|anymore|ever|at\s+all|either|now|today|yet|here|there"
    r"|make|keep|sound|stay|be|use|never|never\s+use|do\s+not\s+use|don'?t\s+use"
    r"|stop\s+using|avoid|switch|change|set|format|write|allow|permit|unban"
    r"|enable|save|remember|store|try|turn|on\s+linkedin|in\s+linkedin|for\s+linkedin)$",
    re.IGNORECASE,
)
_LEADING_WRAPPER_RE = re.compile(
    r"^(?:(?:the|any|all|stiff|corporate|cringe|overused|cliche|cliché|formal|robotic)\s+)*"
    r"(?:(?:words?|terms?|buzzwords?|phrases?|expressions?)(?:\s+(?:like|such\s+as))?(?:\s+|$))?",
    re.IGNORECASE,
)
_META_WRAPPER_TOKEN_RE = re.compile(
    r"\b(?:words?|terms?|phrases?|buzzwords?|emojis?|emoticons?|hashtags?|bullets?|like|such\s+as)\b",
    re.IGNORECASE,
)
_NON_LITERAL_VERB_PREFIX_RE = re.compile(
    r"^(?:being|sounding|making|writing|too|very|long|short|verbose|formal|stiff|robotic|corporate)\b",
    re.IGNORECASE,
)
_CANDIDATE_TOKEN_RE = re.compile(r"^[a-z][a-z0-9\-]{1,28}(?:\s+[a-z][a-z0-9\-]{1,28})?$")

_EMOJI_ALLOW_RE = re.compile(
    r"\b(?:allow|enable|bring\s+back|you\s+can\s+use|include)\s+(?:some\s+)?emojis?\b",
    re.IGNORECASE,
)
_EMOJI_BAN_RE = re.compile(
    r"(?:\b(?:no|zero|without|avoid|strip|remove|drop|ban|skip)\s+(?:any\s+|all\s+)?emojis?\b"
    r"|\b(?:never|don'?t|do\s+not)\s+(?:use|include|add)\b[^.;!]*\bemojis?\b)",
    re.IGNORECASE,
)
_CONCISE_FALSE_RE = re.compile(
    r"\b(?:make\s+it\s+longer|longer|more\s+detailed|multi[- ]paragraph|expand\s+it)\b",
    re.IGNORECASE,
)
_CONCISE_TRUE_RE = re.compile(
    r"\b(?:shorter|concise|brief|punchy|punchier|1[-–]2\s+sentences|one\s+or\s+two\s+sentences|less\s+wordy|trim\s+fluff)\b",
    re.IGNORECASE,
)
_FORMAT_BULLETS_BAN_RE = re.compile(
    r"\b(?:without|no|avoid|remove|drop|skip)\s+(?:any\s+)?(?:bullet\s*points?|bulleted(?:\s+list)?|bullets)\b",
    re.IGNORECASE,
)
_FORMAT_BULLETS_RE = re.compile(
    r"\b(?:bullet\s*points?|bulleted(?:\s+list)?|bullets)\b",
    re.IGNORECASE,
)
_FORMAT_PARAGRAPH_RE = re.compile(
    r"\b(?:paragraphs?|paragraph\s+format|prose\s+format|single\s+paragraph)\b",
    re.IGNORECASE,
)
_FORMAT_BROETRY_RE = re.compile(
    r"\b(?:broetry|one[- ]line\s+stanzas?|spaced\s+single[- ]line)\b",
    re.IGNORECASE,
)
_STYLE_RESET_RE = re.compile(
    r"\b(?:switch|reset|change)\s+(?:my\s+)?(?:default\s+)?style\s+to\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RuntimeConfigSnapshot:
    """Resolved runtime configuration combining M1 settings with live env vars."""

    app_name: str
    data_dir: Path
    session_service_uri: str | None
    google_genai_use_vertexai: bool
    use_vertex_memory_bank: bool
    google_cloud_project: str | None
    google_cloud_location: str
    agent_engine_id: str | None


@dataclass(frozen=True)
class VectorSearchMatch:
    """Ranked semantic ``VectorSearch`` similarity match for an ``IdiomExample``."""

    example: IdiomExample
    similarity_score: float


@dataclass
class PreferenceSlots:
    """Orthogonal style metadata slots preventing stale rule duplication."""

    voice_slot: str = ""
    energy_slot: str = ""
    preferred_phrases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize orthogonal preference slots to a JSON-compatible dict."""
        return {
            "voice_slot": self.voice_slot,
            "energy_slot": self.energy_slot,
            "preferred_phrases": list(self.preferred_phrases),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> PreferenceSlots:
        """Hydrate ``PreferenceSlots`` from serialized JSON metadata."""
        if not payload:
            return cls()
        phrases_raw = payload.get("preferred_phrases", [])
        phrases = [str(item) for item in phrases_raw if str(item).strip()]
        return cls(
            voice_slot=str(payload.get("voice_slot", "")),
            energy_slot=str(payload.get("energy_slot", "")),
            preferred_phrases=phrases,
        )


def _parse_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_runtime_settings() -> RuntimeConfigSnapshot:
    """Resolve M1 ``AppSettings`` dynamically while honoring live env overrides."""
    m1_settings: Any = None
    try:
        from thrilled_to_announce.config.settings import get_settings

        m1_settings = get_settings()
    except Exception:
        m1_settings = None

    env_data_dir = os.getenv("THRILLED_DATA_DIR")
    data_dir = (
        Path(env_data_dir)
        if env_data_dir
        else Path(getattr(m1_settings, "data_dir", ".thrilled_data"))
    )
    session_uri = os.getenv("SESSION_SERVICE_URI") or getattr(
        m1_settings, "session_service_uri", None
    )
    use_vertex = (
        _parse_bool_env("GOOGLE_GENAI_USE_VERTEXAI")
        if "GOOGLE_GENAI_USE_VERTEXAI" in os.environ
        else bool(getattr(m1_settings, "google_genai_use_vertexai", False))
    )
    use_memory_bank = (
        _parse_bool_env("THRILLED_USE_VERTEX_MEMORY_BANK")
        if "THRILLED_USE_VERTEX_MEMORY_BANK" in os.environ
        else bool(getattr(m1_settings, "use_vertex_memory_bank", False))
    )
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") or getattr(
        m1_settings, "google_cloud_project", None
    )
    if project_id and project_id.strip().upper() == "YOUR_PROJECT_ID":
        project_id = None
    location = os.getenv("GOOGLE_CLOUD_LOCATION") or getattr(
        m1_settings, "google_cloud_location", "us-central1"
    )
    agent_engine_id = os.getenv("AGENT_ENGINE_ID") or getattr(
        m1_settings, "agent_engine_id", None
    )
    app_name = str(getattr(m1_settings, "app_name", "thrilled_to_announce"))
    return RuntimeConfigSnapshot(
        app_name=app_name,
        data_dir=data_dir,
        session_service_uri=session_uri,
        google_genai_use_vertexai=use_vertex,
        use_vertex_memory_bank=use_memory_bank,
        google_cloud_project=project_id,
        google_cloud_location=location,
        agent_engine_id=agent_engine_id,
    )


def scrub_pii_text(text: str) -> tuple[str, int, list[str]]:
    """Scrub PII and secret tokens via M1 ``PiiScrubber`` plus regex fallback."""
    if not text:
        return "", 0, []

    current = text
    total_count = 0
    redacted_types: list[str] = []

    try:
        from thrilled_to_announce.observability.pii_scrubber import scrub_text

        m1_res = scrub_text(current)
        current = str(m1_res.scrubbed_text)
        total_count += int(m1_res.redactions_count)
        for item in m1_res.redacted_types:
            if item not in redacted_types:
                redacted_types.append(str(item))
    except Exception:
        pass

    for pii_label, replacement, pattern in _LOCAL_PII_PATTERNS:
        current, hits = pattern.subn(replacement, current)
        if hits > 0:
            total_count += hits
            if pii_label not in redacted_types:
                redacted_types.append(pii_label)

    return current, total_count, redacted_types


def _scrub_structlog_value(value: Any) -> Any:
    if isinstance(value, str):
        scrubbed, _, _ = scrub_pii_text(value)
        return scrubbed
    if isinstance(value, dict):
        return {str(k): _scrub_structlog_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub_structlog_value(item) for item in value]
    return value


def _scrub_structlog_event(
    _logger: Any, _method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return {k: _scrub_structlog_value(v) for k, v in event_dict.items()}


def get_structured_logger(name: str) -> Any:
    """Return M1's structured JSON logger or a stderr JSON ``structlog`` fallback."""
    try:
        from thrilled_to_announce.observability.logging import get_logger

        return get_logger(name)
    except Exception:
        base_logger = structlog.WriteLoggerFactory(file=sys.stderr)()
        return structlog.wrap_logger(
            base_logger,
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                _scrub_structlog_event,
                structlog.processors.JSONRenderer(),
            ],
        ).bind(logger=name)


class DenseVectorSearchIndex:
    """Dense ``float32`` L2-normalized ``VectorSearch`` cosine index for idioms.

    Supports cloud semantic embeddings via ``google.genai`` ``embed_content``
    (``text-embedding-004``, ``output_dimensionality=128``) when Vertex AI or a
    live Gemini API key is active, and deterministic signed ``blake2b`` word
    unigram/bigram + character n-gram embeddings offline. Vector payloads pack
    cleanly to 512-byte ``<128f`` blobs suitable for SQLite, ``pgvector``, or
    ``AlloyDB`` vector columns.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        if dim < 16:
            raise ValueError("Vector dimension must be at least 16.")
        self.dim = dim
        self._pack_fmt = f"<{self.dim}f"

    @staticmethod
    def _l2_normalize(values: Sequence[float]) -> list[float]:
        norm = math.sqrt(sum(float(v) * float(v) for v in values))
        if norm <= 1e-12:
            return [0.0 for _ in values]
        return [float(v) / norm for v in values]

    def _accumulate_feature(
        self, vec: list[float], feature_key: str, weight: float
    ) -> None:
        digest = hashlib.blake2b(feature_key.encode("utf-8"), digest_size=8).digest()
        bucket_raw = int.from_bytes(digest[:4], byteorder="little", signed=False)
        sign_bit = digest[4] & 1
        bucket = bucket_raw % self.dim
        sign = 1.0 if sign_bit == 0 else -1.0
        vec[bucket] += sign * weight

    def embed_text_deterministic(self, text: str) -> list[float]:
        """Compute deterministic L2-normalized ``float32`` n-gram embeddings."""
        scrubbed, _, _ = scrub_pii_text(text)
        normalized = scrubbed.lower().strip()
        vec = [0.0] * self.dim
        if not normalized:
            return vec

        tokens = re.findall(r"[a-z0-9']+", normalized)
        for token in tokens:
            weight = 0.25 if token in _STOP_WORDS else 2.0
            self._accumulate_feature(vec, f"w1:{token}", weight)
            padded = f"<{token}>"
            for n_size in (3, 4):
                if len(padded) >= n_size:
                    for idx in range(len(padded) - n_size + 1):
                        self._accumulate_feature(
                            vec, f"c{n_size}:{padded[idx : idx + n_size]}", 0.75
                        )

        for idx in range(len(tokens) - 1):
            bigram = f"{tokens[idx]}_{tokens[idx + 1]}"
            self._accumulate_feature(vec, f"w2:{bigram}", 2.5)

        if not tokens:
            compact_symbols = re.sub(r"\s+", " ", normalized)
            padded_symbols = f"<{compact_symbols}>"
            for n_size in (2, 3, 4):
                if len(padded_symbols) >= n_size:
                    for idx in range(len(padded_symbols) - n_size + 1):
                        self._accumulate_feature(
                            vec,
                            f"sym{n_size}:{padded_symbols[idx : idx + n_size]}",
                            1.0,
                        )

        if sum(float(v) * float(v) for v in vec) <= 1e-12:
            vec[0] = 1.0

        packed = self.pack_vector(self._l2_normalize(vec))
        return self.unpack_vector(packed)

    async def embed_text_async(self, text: str) -> list[float]:
        """Compute L2-normalized embedding via ``genai`` cloud API or local fallback."""
        scrubbed, _, _ = scrub_pii_text(text)
        if not scrubbed.strip():
            return [0.0] * self.dim

        cfg = get_runtime_settings()
        raw_key = os.getenv("GEMINI_API_KEY", "").strip()
        is_placeholder_key = raw_key.lower().startswith(
            ("test_", "dummy_", "fake_", "placeholder_", "your_")
        )
        can_use_vertex = bool(
            cfg.google_genai_use_vertexai and cfg.google_cloud_project
        )
        can_use_api_key = bool(raw_key and not is_placeholder_key)

        if can_use_vertex or can_use_api_key:
            try:
                client = (
                    genai.Client(
                        vertexai=True,
                        project=cfg.google_cloud_project,
                        location=cfg.google_cloud_location,
                    )
                    if can_use_vertex
                    else genai.Client(api_key=raw_key)
                )
                response = await client.aio.models.embed_content(
                    model="text-embedding-004",
                    contents=scrubbed,
                    config=types.EmbedContentConfig(
                        task_type="SEMANTIC_SIMILARITY",
                        output_dimensionality=self.dim,
                    ),
                )
                if response.embeddings and response.embeddings[0].values:
                    raw_vals = list(response.embeddings[0].values)
                    if len(raw_vals) == self.dim:
                        packed = self.pack_vector(self._l2_normalize(raw_vals))
                        return self.unpack_vector(packed)
            except Exception as exc:
                get_structured_logger(__name__).debug(
                    "embedding_cloud_fallback",
                    reason=str(exc),
                )

        return self.embed_text_deterministic(scrubbed)

    def pack_vector(self, vector: Sequence[float]) -> bytes:
        """Pack a ``dim``-length float sequence into little-endian ``float32`` bytes."""
        if len(vector) != self.dim:
            raise ValueError(f"Expected vector of length {self.dim}, got {len(vector)}.")
        return struct.pack(self._pack_fmt, *[float(x) for x in vector])

    def unpack_vector(self, blob: bytes) -> list[float]:
        """Unpack little-endian ``float32`` bytes into a Python ``list[float]``."""
        expected_bytes = self.dim * 4
        if len(blob) != expected_bytes:
            raise ValueError(
                f"Expected {expected_bytes} embedding bytes, got {len(blob)}."
            )
        return [float(x) for x in struct.unpack(self._pack_fmt, blob)]

    @staticmethod
    def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
        """Compute cosine similarity between two L2-normalized vectors."""
        if not vec_a or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
        return max(-1.0, min(1.0, dot))

    def rank_top_k(
        self,
        query_vec: Sequence[float],
        candidates: Sequence[tuple[IdiomExample, Sequence[float]]],
        top_k: int = 3,
    ) -> list[VectorSearchMatch]:
        """Return top-k ``VectorSearchMatch`` entries ranked by cosine similarity."""
        scored: list[VectorSearchMatch] = []
        has_query_signal = any(abs(v) > 1e-9 for v in query_vec)
        for example, cand_vec in candidates:
            score = (
                round(self.cosine_similarity(query_vec, cand_vec), 4)
                if has_query_signal
                else 0.0
            )
            updated_example = example.model_copy(update={"similarity_score": score})
            scored.append(
                VectorSearchMatch(example=updated_example, similarity_score=score)
            )
        scored.sort(key=lambda item: item.similarity_score, reverse=True)
        return scored[: max(1, top_k)]


def normalize_preferred_format(
    raw_format: PreferredFormat | str | None,
) -> tuple[PreferredFormat | None, bool]:
    """Normalize caller format strings and map ``short_punchy`` to ``keep_concise``."""
    if raw_format is None:
        return None, False
    if isinstance(raw_format, PreferredFormat):
        return raw_format, False

    cleaned = raw_format.strip().lower().replace("-", "_").replace(" ", "_")
    if not cleaned:
        return None, False
    if cleaned in {"paragraph", "paragraphs", "prose"}:
        return PreferredFormat.PARAGRAPH, False
    if cleaned in {"bullets", "bullet", "bullet_points", "bullet_point"}:
        return PreferredFormat.BULLETS, False
    if cleaned in {"broetry", "spaced_lines", "one_line_stanzas"}:
        return PreferredFormat.BROETRY, False
    if cleaned in {"short_punchy", "punchy", "concise", "short", "brief"}:
        return PreferredFormat.PARAGRAPH, True

    raise ValueError(
        f"Unsupported preferred_format '{raw_format}'. Valid options are "
        "'paragraph', 'bullets', 'broetry', or 'short_punchy'. "
        "Please check preferred_format and try again."
    )


def _normalize_quotes(text: str) -> str:
    return (
        text.replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def _mask_quoted_spans(text: str) -> str:
    return _QUOTED_ITEM_RE.sub(lambda m: " " * len(m.group(0)), text)


def _truncate_at_directive_boundary(clause: str) -> str:
    masked = _mask_quoted_spans(clause)
    boundary_match = _CLAUSE_DIRECTIVE_BOUNDARY_RE.search(masked)
    if boundary_match is not None:
        return clause[: boundary_match.start()]
    return clause


def _strip_trailing_modifiers(fragment: str) -> str:
    cleaned = fragment.strip(" ,;:-—.!?")
    while cleaned:
        updated = _TRAILING_CLAUSE_CONTEXT_RE.sub("", cleaned).strip(" ,;:-—.!?")
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def _extract_clause_words(clause: str, *, quoted_only: bool = False) -> list[str]:
    extracted: list[str] = []
    normalized = _truncate_at_directive_boundary(_normalize_quotes(clause))

    for match in _QUOTED_ITEM_RE.finditer(normalized):
        candidate = re.sub(r"\s+", " ", match.group(1).strip().lower()).strip(".,;:!?")
        if (
            candidate
            and candidate not in META_CATEGORY_TOKENS
            and not _META_WRAPPER_TOKEN_RE.fullmatch(candidate)
            and not _STANDALONE_DIRECTIVE_OR_MODIFIER_RE.fullmatch(candidate)
            and candidate not in extracted
        ):
            extracted.append(candidate)

    if quoted_only:
        return extracted

    unquoted = _QUOTED_ITEM_RE.sub(" ", normalized)
    unquoted = _truncate_at_directive_boundary(unquoted)
    unquoted = _strip_trailing_modifiers(unquoted)
    unquoted = _LEADING_WRAPPER_RE.sub("", unquoted).strip(" ,;:-—.!?")
    unquoted = _strip_trailing_modifiers(unquoted)
    if not unquoted:
        return extracted

    parts = re.split(r"\s*(?:,|\bor\b|\band\b)\s*", unquoted, flags=re.IGNORECASE)
    for part in parts:
        token = _LEADING_BAN_VERB_RE.sub("", part.strip(" ,;:-—.!?")).strip()
        token = _LEADING_WRAPPER_RE.sub("", token).strip(" ,;:-—.!?")
        token = _strip_trailing_modifiers(token).lower()
        if not token or token in META_CATEGORY_TOKENS:
            continue
        if _STANDALONE_DIRECTIVE_OR_MODIFIER_RE.fullmatch(token):
            continue
        if _CLAUSE_DIRECTIVE_BOUNDARY_RE.search(token):
            continue
        if _META_WRAPPER_TOKEN_RE.search(token):
            continue
        if _NON_LITERAL_VERB_PREFIX_RE.search(token):
            continue
        if _CANDIDATE_TOKEN_RE.fullmatch(token) and token not in extracted:
            extracted.append(token)
    return extracted


def _iter_verb_subclauses(
    instruction: str, verb_head_re: re.Pattern[str]
) -> list[str]:
    masked = _mask_quoted_spans(instruction)
    matches = list(verb_head_re.finditer(masked))
    subclauses: list[str] = []
    for idx, match in enumerate(matches):
        start = match.end()
        limit = matches[idx + 1].start() if idx + 1 < len(matches) else len(instruction)
        punct_stop = re.search(r"[.;—!]", masked[start:limit])
        end = start + punct_stop.start() if punct_stop is not None else limit
        subclauses.append(instruction[start:end])
    return subclauses


def _infer_slots_from_summary(tone_summary: str) -> PreferenceSlots:
    lower = tone_summary.lower()
    voice = ""
    if "more human" in lower:
        voice = "more human"
    elif "conversational and natural" in lower:
        voice = "conversational and natural"
    elif "direct and pragmatic" in lower:
        voice = "direct and pragmatic"
    elif "formal executive" in lower:
        voice = "formal executive"

    energy = ""
    if "upbeat" in lower:
        energy = "upbeat"
    elif "understated" in lower:
        energy = "understated"

    return PreferenceSlots(voice_slot=voice, energy_slot=energy)


def parse_and_merge_preferences(
    existing: WordingPreferenceProfile,
    raw_instruction: str,
    *,
    existing_slots: PreferenceSlots | None = None,
    banned_words: Sequence[str] | None = None,
    preferred_phrases: Sequence[str] | None = None,
    avoid_emojis: bool | None = None,
    keep_concise: bool | None = None,
    preferred_format: PreferredFormat | str | None = None,
) -> tuple[WordingPreferenceProfile, PreferenceSlots]:
    """Parse natural-language style tweaks and cleanly merge into ``existing``."""
    scrubbed_instr, _, _ = scrub_pii_text(_normalize_quotes(raw_instruction).strip())
    slots = (
        PreferenceSlots(
            voice_slot=existing_slots.voice_slot,
            energy_slot=existing_slots.energy_slot,
            preferred_phrases=list(existing_slots.preferred_phrases),
        )
        if existing_slots is not None
        else _infer_slots_from_summary(existing.tone_summary)
    )

    is_style_reset = bool(_STYLE_RESET_RE.search(scrubbed_instr))
    if is_style_reset:
        slots.voice_slot = ""
        slots.energy_slot = ""

    explicit_fmt, concise_from_fmt = normalize_preferred_format(preferred_format)

    unbanned_set: set[str] = set()
    for unban_subclause in _iter_verb_subclauses(scrubbed_instr, _UNBAN_VERB_HEAD_RE):
        for word in _extract_clause_words(unban_subclause):
            unbanned_set.add(word)
    for unban_rm_match in _UNBAN_FROM_BANNED_RE.finditer(scrubbed_instr):
        for word in _extract_clause_words(unban_rm_match.group(1)):
            unbanned_set.add(word)

    merged_banned: list[str] = [
        bw for bw in existing.banned_words if bw.lower() not in unbanned_set
    ]
    for ban_subclause in _iter_verb_subclauses(scrubbed_instr, _BAN_VERB_HEAD_RE):
        for word in _extract_clause_words(ban_subclause):
            if word not in merged_banned and word not in unbanned_set:
                merged_banned.append(word)
    for rm_match in _REMOVE_QUOTED_BAN_RE.finditer(scrubbed_instr):
        rm_clause = rm_match.group(1)
        if _FROM_BANNED_LIST_GUARD_RE.search(rm_clause):
            continue
        for word in _extract_clause_words(rm_clause, quoted_only=True):
            if word not in merged_banned and word not in unbanned_set:
                merged_banned.append(word)

    if banned_words:
        for raw_bw in banned_words:
            scrubbed_bw, _, _ = scrub_pii_text(str(raw_bw).strip().lower())
            cleaned_bw = scrubbed_bw.strip(" '\".,;:!?")
            if (
                cleaned_bw
                and cleaned_bw not in META_CATEGORY_TOKENS
                and cleaned_bw not in merged_banned
            ):
                merged_banned.append(cleaned_bw)

    if avoid_emojis is not None:
        resolved_avoid_emojis = bool(avoid_emojis)
    elif _EMOJI_ALLOW_RE.search(scrubbed_instr):
        resolved_avoid_emojis = False
    elif _EMOJI_BAN_RE.search(scrubbed_instr):
        resolved_avoid_emojis = True
    else:
        resolved_avoid_emojis = existing.avoid_emojis

    if keep_concise is not None:
        resolved_concise = bool(keep_concise)
    elif concise_from_fmt:
        resolved_concise = True
    elif _CONCISE_FALSE_RE.search(scrubbed_instr):
        resolved_concise = False
    elif _CONCISE_TRUE_RE.search(scrubbed_instr):
        resolved_concise = True
    elif is_style_reset:
        resolved_concise = False
    else:
        resolved_concise = existing.keep_concise

    if explicit_fmt is not None:
        resolved_format = explicit_fmt
    elif _FORMAT_BULLETS_BAN_RE.search(scrubbed_instr) or _FORMAT_PARAGRAPH_RE.search(
        scrubbed_instr
    ):
        resolved_format = PreferredFormat.PARAGRAPH
    elif _FORMAT_BULLETS_RE.search(scrubbed_instr):
        resolved_format = PreferredFormat.BULLETS
    elif _FORMAT_BROETRY_RE.search(scrubbed_instr):
        resolved_format = PreferredFormat.BROETRY
    else:
        resolved_format = existing.preferred_format

    lower_instr = scrubbed_instr.lower()
    if re.search(r"\b(?:more\s+human|sound\s+human|human[- ]sounding)\b", lower_instr):
        slots.voice_slot = "more human"
    elif re.search(
        r"\b(?:less\s+corporate|less\s+stiff|less\s+robotic|conversational|natural|more\s+casual)\b",
        lower_instr,
    ):
        slots.voice_slot = "conversational and natural"
    elif re.search(r"\b(?:direct|pragmatic|no[- ]nonsense|candid)\b", lower_instr):
        slots.voice_slot = "direct and pragmatic"
    elif re.search(r"\b(?:more\s+formal|executive\s+tone|boardroom)\b", lower_instr):
        slots.voice_slot = "formal executive"

    if re.search(r"\b(?:upbeat|energetic|enthusiastic|warmer|warm)\b", lower_instr):
        slots.energy_slot = "upbeat"
    elif re.search(r"\b(?:understated|calm|reserved|measured)\b", lower_instr):
        slots.energy_slot = "understated"

    if preferred_phrases:
        for phrase in preferred_phrases:
            scrubbed_phrase, _, _ = scrub_pii_text(str(phrase).strip())
            if scrubbed_phrase and scrubbed_phrase not in slots.preferred_phrases:
                slots.preferred_phrases.append(scrubbed_phrase)

    summary_parts: list[str] = []
    if resolved_concise:
        summary_parts.append("concise")
    if slots.energy_slot:
        summary_parts.append(slots.energy_slot)
    if slots.voice_slot:
        summary_parts.append(slots.voice_slot)
    if resolved_format == PreferredFormat.BULLETS:
        summary_parts.append("bullet-point format")
    elif resolved_format == PreferredFormat.BROETRY:
        summary_parts.append("spaced single-line format")

    tone_summary = (
        ", ".join(summary_parts) if summary_parts else "clean professional polish"
    )
    if slots.preferred_phrases:
        joined_phrases = "; ".join(slots.preferred_phrases[:5])
        tone_summary = f"{tone_summary} (preferred phrases: {joined_phrases})"
    tone_summary, _, _ = scrub_pii_text(tone_summary)

    updated = WordingPreferenceProfile(
        user_id=existing.user_id,
        tone_summary=tone_summary,
        banned_words=merged_banned,
        avoid_emojis=resolved_avoid_emojis,
        keep_concise=resolved_concise,
        preferred_format=resolved_format,
        has_custom_preferences=True,
    )
    return updated, slots


_GLOBAL_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()


class VectorStyleMemoryStore:
    """Persistent SQLite and dense vector store for user wording preferences."""

    def __init__(
        self,
        data_dir: Path | str | None = None,
        vector_index: DenseVectorSearchIndex | None = None,
    ) -> None:
        cfg = get_runtime_settings()
        base_dir = Path(data_dir) if data_dir is not None else cfg.data_dir
        self.data_dir: Path = base_dir.resolve()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sqlite_path: Path = (self.data_dir / "style_memory.sqlite3").resolve()
        self.vector_index: DenseVectorSearchIndex = (
            vector_index or DenseVectorSearchIndex(dim=EMBEDDING_DIM)
        )
        self._profiles_cache: dict[str, WordingPreferenceProfile] = {}
        self._slots_cache: dict[str, PreferenceSlots] = {}
        self._vectors_cache: dict[str, list[tuple[IdiomExample, list[float]]]] = {}
        self._hydrated_users: set[str] = set()
        self._pending_publishes_cache: dict[str, dict[str, Any]] = {}
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._persist_locks: dict[int, asyncio.Lock] = {}
        self._schema_initialized: bool = False
        self._ensure_schema_sync()

    def _get_persist_lock(self) -> asyncio.Lock:
        """Return an event-loop-scoped ``asyncio.Lock`` serializing SQLite writes."""
        loop_key = id(asyncio.get_running_loop())
        lock = self._persist_locks.get(loop_key)
        if lock is None:
            lock = asyncio.Lock()
            self._persist_locks[loop_key] = lock
        return lock

    def _ensure_schema_sync(self) -> None:
        if self._schema_initialized and self.sqlite_path.exists():
            return
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self.sqlite_path), timeout=30.0) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    slots_json TEXT NOT NULL DEFAULT '{}',
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS idiom_vectors (
                    entry_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    human_text TEXT NOT NULL,
                    linkedin_text TEXT NOT NULL,
                    dim INTEGER NOT NULL,
                    embedding_blob BLOB NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_idiom_vectors_user
                ON idiom_vectors (user_id, created_at DESC)
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS pending_publishes (
                    publish_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    post_content TEXT NOT NULL,
                    preview_text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.commit()
        self._schema_initialized = True

    def _hydrate_user_from_sqlite_sync(self, user_id: str) -> None:
        if user_id in self._hydrated_users:
            return
        self._ensure_schema_sync()
        with sqlite3.connect(str(self.sqlite_path), timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=10000;")
            row = conn.execute(
                "SELECT profile_json, slots_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row is not None:
                profile_data = json.loads(str(row[0]))
                slots_data = json.loads(str(row[1])) if row[1] else {}
                self._profiles_cache[user_id] = WordingPreferenceProfile.model_validate(
                    profile_data
                )
                self._slots_cache[user_id] = PreferenceSlots.from_dict(slots_data)

            vec_rows = conn.execute(
                """
                SELECT entry_id, user_id, human_text, linkedin_text, dim, embedding_blob
                FROM idiom_vectors
                WHERE user_id = ?
                ORDER BY created_at DESC
                """,
                (user_id,),
            ).fetchall()
            loaded_vectors: list[tuple[IdiomExample, list[float]]] = []
            for entry_id, r_uid, h_text, l_text, dim, blob in vec_rows:
                if int(dim) == self.vector_index.dim:
                    vec = self.vector_index.unpack_vector(bytes(blob))
                else:
                    vec = self.vector_index.embed_text_deterministic(
                        f"{h_text} {l_text}"
                    )
                example = IdiomExample(
                    entry_id=str(entry_id),
                    user_id=str(r_uid),
                    human_text=str(h_text),
                    linkedin_text=str(l_text),
                    similarity_score=0.0,
                )
                loaded_vectors.append((example, vec))
            self._vectors_cache[user_id] = loaded_vectors
        self._hydrated_users.add(user_id)

    def get_profile(self, user_id: str = "default_user") -> WordingPreferenceProfile:
        """Return the user's ``WordingPreferenceProfile`` with read-through hydration."""
        self._hydrate_user_from_sqlite_sync(user_id)
        if user_id in self._profiles_cache:
            return self._profiles_cache[user_id].model_copy(deep=True)
        return WordingPreferenceProfile(user_id=user_id)

    async def get_profile_async(
        self, user_id: str = "default_user"
    ) -> WordingPreferenceProfile:
        """Async ``aiosqlite`` read-through lookup for ``WordingPreferenceProfile``."""
        if user_id in self._profiles_cache:
            return self._profiles_cache[user_id].model_copy(deep=True)
        self._ensure_schema_sync()
        async with aiosqlite.connect(str(self.sqlite_path), timeout=30.0) as conn:
            await conn.execute("PRAGMA busy_timeout=10000;")
            async with conn.execute(
                "SELECT profile_json, slots_json FROM user_profiles WHERE user_id = ?",
                (user_id,),
            ) as cursor:
                row = await cursor.fetchone()
                if row is not None:
                    profile = WordingPreferenceProfile.model_validate(
                        json.loads(str(row[0]))
                    )
                    self._profiles_cache[user_id] = profile
                    self._slots_cache[user_id] = PreferenceSlots.from_dict(
                        json.loads(str(row[1])) if row[1] else {}
                    )
                    return profile.model_copy(deep=True)
        return WordingPreferenceProfile(user_id=user_id)

    def search_similar_examples(
        self,
        user_id: str = "default_user",
        query_text: str = "",
        top_k: int = 3,
    ) -> list[IdiomExample]:
        """Search user ``IdiomExample`` entries by dense cosine similarity."""
        self._hydrate_user_from_sqlite_sync(user_id)
        candidates = self._vectors_cache.get(user_id, [])
        if not candidates:
            return []
        scrubbed_query, _, _ = scrub_pii_text(query_text)
        query_vec = (
            self.vector_index.embed_text_deterministic(scrubbed_query)
            if scrubbed_query.strip()
            else [0.0] * self.vector_index.dim
        )
        matches = self.vector_index.rank_top_k(query_vec, candidates, top_k=top_k)
        return [match.example for match in matches]

    async def search_similar_examples_async(
        self,
        user_id: str = "default_user",
        query_text: str = "",
        top_k: int = 3,
    ) -> list[IdiomExample]:
        """Async semantic top-k search across ``aiosqlite``-stored ``IdiomExample`` rows."""
        if user_id not in self._hydrated_users:
            self._ensure_schema_sync()
            async with aiosqlite.connect(
                str(self.sqlite_path), timeout=30.0
            ) as conn:
                await conn.execute("PRAGMA busy_timeout=10000;")
                async with conn.execute(
                    """
                    SELECT entry_id, user_id, human_text, linkedin_text, dim, embedding_blob
                    FROM idiom_vectors
                    WHERE user_id = ?
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                ) as cursor:
                    rows = await cursor.fetchall()
            loaded: list[tuple[IdiomExample, list[float]]] = []
            for entry_id, r_uid, h_text, l_text, dim, blob in rows:
                vec = (
                    self.vector_index.unpack_vector(bytes(blob))
                    if int(dim) == self.vector_index.dim
                    else await self.vector_index.embed_text_async(f"{h_text} {l_text}")
                )
                loaded.append(
                    (
                        IdiomExample(
                            entry_id=str(entry_id),
                            user_id=str(r_uid),
                            human_text=str(h_text),
                            linkedin_text=str(l_text),
                            similarity_score=0.0,
                        ),
                        vec,
                    )
                )
            self._vectors_cache[user_id] = loaded
            await self.get_profile_async(user_id)
            self._hydrated_users.add(user_id)

        candidates = self._vectors_cache.get(user_id, [])
        if not candidates:
            return []
        scrubbed_query, _, _ = scrub_pii_text(query_text)
        query_vec = (
            await self.vector_index.embed_text_async(scrubbed_query)
            if scrubbed_query.strip()
            else [0.0] * self.vector_index.dim
        )
        matches = self.vector_index.rank_top_k(query_vec, candidates, top_k=top_k)
        return [match.example for match in matches]

    def _build_new_examples(
        self,
        user_id: str,
        *,
        preferred_phrases: Sequence[str] | None = None,
        approved_human_example: str | None = None,
        approved_linkedin_example: str | None = None,
    ) -> list[IdiomExample]:
        new_examples: list[IdiomExample] = []
        if approved_human_example or approved_linkedin_example:
            scrub_h, _, _ = scrub_pii_text(approved_human_example or "")
            scrub_l, _, _ = scrub_pii_text(approved_linkedin_example or "")
            if scrub_h.strip() or scrub_l.strip():
                new_examples.append(
                    IdiomExample(
                        entry_id=f"idiom_{uuid.uuid4().hex[:12]}",
                        user_id=user_id,
                        human_text=scrub_h.strip() or scrub_l.strip(),
                        linkedin_text=scrub_l.strip() or scrub_h.strip(),
                        similarity_score=0.0,
                    )
                )

        if preferred_phrases:
            for phrase in preferred_phrases:
                scrub_p, _, _ = scrub_pii_text(str(phrase).strip())
                if scrub_p:
                    new_examples.append(
                        IdiomExample(
                            entry_id=f"idiom_{uuid.uuid4().hex[:12]}",
                            user_id=user_id,
                            human_text=scrub_p,
                            linkedin_text=scrub_p,
                            similarity_score=0.0,
                        )
                    )
        return new_examples

    def _persist_consolidation_sync(
        self,
        profile: WordingPreferenceProfile,
        slots: PreferenceSlots,
        examples_with_vecs: Sequence[tuple[IdiomExample, list[float]]],
    ) -> None:
        self._ensure_schema_sync()
        now = time.time()
        with sqlite3.connect(str(self.sqlite_path), timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(
                """
                INSERT INTO user_profiles (user_id, profile_json, slots_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    slots_json = excluded.slots_json,
                    updated_at = excluded.updated_at
                """,
                (
                    profile.user_id,
                    json.dumps(profile.model_dump(mode="json")),
                    json.dumps(slots.to_dict()),
                    now,
                ),
            )
            for example, vec in examples_with_vecs:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO idiom_vectors (
                        entry_id, user_id, human_text, linkedin_text, dim, embedding_blob, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        example.entry_id,
                        example.user_id,
                        example.human_text,
                        example.linkedin_text,
                        self.vector_index.dim,
                        self.vector_index.pack_vector(vec),
                        now,
                    ),
                )
            conn.commit()

    async def _persist_consolidation_async(
        self,
        profile: WordingPreferenceProfile,
        slots: PreferenceSlots,
        examples: Sequence[IdiomExample],
    ) -> None:
        user_id = profile.user_id
        try:
            embedded_rows: list[tuple[IdiomExample, list[float]]] = []
            for ex in examples:
                vec = await self.vector_index.embed_text_async(
                    f"{ex.human_text} {ex.linkedin_text}"
                )
                embedded_rows.append((ex, vec))

            async with self._get_persist_lock():
                self._ensure_schema_sync()
                now = time.time()
                user_vecs = self._vectors_cache.setdefault(user_id, [])
                existing_ids = {item[0].entry_id for item in user_vecs}
                for ex, vec in embedded_rows:
                    if ex.entry_id not in existing_ids:
                        user_vecs.insert(0, (ex, vec))
                    else:
                        for idx, (cur_ex, _) in enumerate(user_vecs):
                            if cur_ex.entry_id == ex.entry_id:
                                user_vecs[idx] = (ex, vec)
                                break

                latest_profile = self._profiles_cache.get(user_id, profile)
                latest_slots = self._slots_cache.get(user_id, slots)

                async with aiosqlite.connect(
                    str(self.sqlite_path), timeout=30.0
                ) as conn:
                    await conn.execute("PRAGMA busy_timeout=10000;")
                    await conn.execute(
                        """
                        INSERT INTO user_profiles (user_id, profile_json, slots_json, updated_at)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT(user_id) DO UPDATE SET
                            profile_json = excluded.profile_json,
                            slots_json = excluded.slots_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            user_id,
                            json.dumps(latest_profile.model_dump(mode="json")),
                            json.dumps(latest_slots.to_dict()),
                            now,
                        ),
                    )
                    for example, vec in embedded_rows:
                        await conn.execute(
                            """
                            INSERT OR REPLACE INTO idiom_vectors (
                                entry_id, user_id, human_text, linkedin_text, dim, embedding_blob, created_at
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                example.entry_id,
                                example.user_id,
                                example.human_text,
                                example.linkedin_text,
                                self.vector_index.dim,
                                self.vector_index.pack_vector(vec),
                                now,
                            ),
                        )
                    await conn.commit()

            get_structured_logger(__name__).info(
                "style_preference_persisted",
                user_id=user_id,
                examples_added=len(embedded_rows),
            )
        except Exception as exc:
            scrubbed_err, _, _ = scrub_pii_text(str(exc))
            get_structured_logger(__name__).warning(
                "style_preference_persistence_failed",
                user_id=user_id,
                error_message=scrubbed_err,
            )

    def schedule_async_consolidation(
        self,
        *,
        user_id: str,
        raw_instruction: str,
        banned_words: Sequence[str] | None = None,
        preferred_phrases: Sequence[str] | None = None,
        avoid_emojis: bool | None = None,
        keep_concise: bool | None = None,
        preferred_format: PreferredFormat | str | None = None,
        approved_human_example: str | None = None,
        approved_linkedin_example: str | None = None,
    ) -> tuple[
        WordingPreferenceProfile,
        Literal["async_background_queued", "synchronous_fallback"],
    ]:
        """Consolidate wording preferences immediately and persist non-blockingly."""
        self._hydrate_user_from_sqlite_sync(user_id)
        existing_profile = self._profiles_cache.get(
            user_id, WordingPreferenceProfile(user_id=user_id)
        )
        existing_slots = self._slots_cache.get(user_id)

        updated_profile, updated_slots = parse_and_merge_preferences(
            existing_profile,
            raw_instruction,
            existing_slots=existing_slots,
            banned_words=banned_words,
            preferred_phrases=preferred_phrases,
            avoid_emojis=avoid_emojis,
            keep_concise=keep_concise,
            preferred_format=preferred_format,
        )
        self._profiles_cache[user_id] = updated_profile
        self._slots_cache[user_id] = updated_slots

        new_examples = self._build_new_examples(
            user_id,
            preferred_phrases=preferred_phrases,
            approved_human_example=approved_human_example,
            approved_linkedin_example=approved_linkedin_example,
        )
        det_embedded: list[tuple[IdiomExample, list[float]]] = [
            (
                ex,
                self.vector_index.embed_text_deterministic(
                    f"{ex.human_text} {ex.linkedin_text}"
                ),
            )
            for ex in new_examples
        ]
        user_vecs = self._vectors_cache.setdefault(user_id, [])
        for ex, vec in reversed(det_embedded):
            user_vecs.insert(0, (ex, vec))

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is not None:
            task = running_loop.create_task(
                self._persist_consolidation_async(
                    updated_profile.model_copy(deep=True),
                    PreferenceSlots.from_dict(updated_slots.to_dict()),
                    new_examples,
                )
            )
            self._background_tasks.add(task)
            _GLOBAL_BACKGROUND_TASKS.add(task)
            task.add_done_callback(self._background_tasks.discard)
            task.add_done_callback(_GLOBAL_BACKGROUND_TASKS.discard)
            return updated_profile.model_copy(deep=True), "async_background_queued"

        self._persist_consolidation_sync(updated_profile, updated_slots, det_embedded)
        return updated_profile.model_copy(deep=True), "synchronous_fallback"

    async def drain_background_tasks(self) -> None:
        """Await all pending background persistence and embedding tasks."""
        while self._background_tasks:
            pending = list(self._background_tasks)
            results = await asyncio.gather(*pending, return_exceptions=True)
            for item in results:
                if isinstance(item, Exception):
                    scrubbed_err, _, _ = scrub_pii_text(str(item))
                    get_structured_logger(__name__).warning(
                        "background_style_task_failed",
                        error_message=scrubbed_err,
                    )

    def save_pending_publish(
        self,
        *,
        publish_id: str,
        user_id: str,
        post_content: str,
        preview_text: str,
        status: str,
    ) -> dict[str, Any]:
        """Persist a HITL pending or finalized LinkedIn publish record to SQLite."""
        scrubbed_uid, _, _ = scrub_pii_text(user_id)
        scrubbed_content, _, _ = scrub_pii_text(post_content)
        scrubbed_preview, _, _ = scrub_pii_text(preview_text)
        now = time.time()
        record = {
            "publish_id": publish_id,
            "user_id": scrubbed_uid,
            "post_content": scrubbed_content,
            "preview_text": scrubbed_preview,
            "status": status,
            "updated_at": now,
        }
        self._pending_publishes_cache[publish_id] = record
        self._ensure_schema_sync()
        with sqlite3.connect(str(self.sqlite_path), timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=10000;")
            conn.execute(
                """
                INSERT INTO pending_publishes (
                    publish_id, user_id, post_content, preview_text, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(publish_id) DO UPDATE SET
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (
                    publish_id,
                    scrubbed_uid,
                    scrubbed_content,
                    scrubbed_preview,
                    status,
                    now,
                ),
            )
            conn.commit()
        return dict(record)

    def get_pending_publish(self, publish_id: str) -> dict[str, Any] | None:
        """Load a HITL publish record from read-through cache or SQLite."""
        if publish_id in self._pending_publishes_cache:
            return dict(self._pending_publishes_cache[publish_id])
        self._ensure_schema_sync()
        with sqlite3.connect(str(self.sqlite_path), timeout=30.0) as conn:
            conn.execute("PRAGMA busy_timeout=10000;")
            row = conn.execute(
                """
                SELECT publish_id, user_id, post_content, preview_text, status, updated_at
                FROM pending_publishes
                WHERE publish_id = ?
                """,
                (publish_id,),
            ).fetchone()
            if row is None:
                return None
            record = {
                "publish_id": str(row[0]),
                "user_id": str(row[1]),
                "post_content": str(row[2]),
                "preview_text": str(row[3]),
                "status": str(row[4]),
                "updated_at": float(row[5]),
            }
            self._pending_publishes_cache[publish_id] = record
            return dict(record)


_STORE_REGISTRY: dict[str, VectorStyleMemoryStore] = {}


def get_default_style_store(
    data_dir: Path | str | None = None,
) -> VectorStyleMemoryStore:
    """Return a cached ``VectorStyleMemoryStore`` keyed by resolved ``data_dir``."""
    cfg = get_runtime_settings()
    target_dir = (Path(data_dir) if data_dir is not None else cfg.data_dir).resolve()
    cache_key = str(target_dir)
    store = _STORE_REGISTRY.get(cache_key)
    if store is None:
        store = VectorStyleMemoryStore(data_dir=target_dir)
        _STORE_REGISTRY[cache_key] = store
    return store


async def drain_all_style_tasks() -> None:
    """Await all global style-consolidation tasks across active store instances."""
    while _GLOBAL_BACKGROUND_TASKS:
        pending = list(_GLOBAL_BACKGROUND_TASKS)
        results = await asyncio.gather(*pending, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                scrubbed_err, _, _ = scrub_pii_text(str(item))
                get_structured_logger(__name__).warning(
                    "global_style_task_failed",
                    error_message=scrubbed_err,
                )
