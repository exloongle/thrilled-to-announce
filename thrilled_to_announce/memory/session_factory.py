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
"""Persistent ADK session and long-term vector memory service factory.

Configures multi-backend persistent session storage (``VertexAiSessionService``
for managed Agent Engine deployments, ``DatabaseSessionService`` for Cloud SQL /
AlloyDB ``postgresql+asyncpg://`` or ``sqlite:///`` URIs when SQLAlchemy is
installed, and ``SqliteSessionService`` backed by ``aiosqlite`` for durable
local/container persistence) alongside ``PersistentVectorMemoryService`` and
``VertexAiMemoryBankService`` with non-blocking ``after_agent_callback`` memory
ingestion.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
from typing import Any

from google.adk.events import Event
from google.adk.memory import (
    BaseMemoryService,
    SqliteMemoryService,
    VertexAiMemoryBankService,
)
from google.adk.memory.base_memory_service import SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions import BaseSessionService, Session, VertexAiSessionService
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools import load_memory, preload_memory
from google.adk.tools.preload_memory_tool import PreloadMemoryTool, preload_memory_tool
from google.genai import types

try:
    from google.adk.sessions import DatabaseSessionService
except ImportError:
    DatabaseSessionService = None  # type: ignore[assignment,misc]

from thrilled_to_announce.memory.style_store import (
    VectorStyleMemoryStore,
    drain_all_style_tasks,
    get_default_style_store,
    get_runtime_settings,
    get_structured_logger,
    scrub_pii_text,
)

_BACKGROUND_MEMORY_TASKS: set[asyncio.Task[Any]] = set()

_SESSION_TWEAK_CUE_RE = re.compile(
    r"(?:"
    r"\b(?:save|remember|store|keep)\s+(?:this|that)\s+(?:preference|rule|style|setting|for\s+next\s+time)\b"
    r"|\b(?:from\s+now\s+on|going\s+forward|for\s+future\s+(?:posts?|drafts?|conversions?))\b"
    r"|\b(?:never|don'?t|do\s+not)\s+(?:use|include)\b"
    r"|\b(?:allow|permit|unban|you\s+can\s+use|okay\s+to\s+use)\b.*(?:\bemojis?\b|['\u2018\u2019\"\u201c\u201d][a-zA-Z])"
    r"|\b(?:allow|enable|disable|strip|remove)\s+(?:all\s+|some\s+)?emojis?\b"
    r"|\b(?:remove|drop|delete|cut|strip)\s+['\u2018\u2019\"\u201c\u201d][a-zA-Z]"
    r"|\b(?:make|keep)\s+(?:it|this|my\s+(?:posts?|drafts?|style))\s+(?:shorter|longer|more\s+detailed|concise|brief)\b"
    r"|\b(?:format\s+(?:it\s+)?(?:as|with)|switch\s+(?:back\s+)?to)\s+(?:upbeat\s+)?(?:bullet\s*points?|bullets|short\s+paragraphs?|paragraph\s+format|paragraphs?|broetry)\b"
    r"|\b(?:switch|change|set|update|reset)\s+(?:my\s+)?(?:default\s+)?(?:style|tone|format|voice|preferences?)\s+to\b"
    r")",
    re.IGNORECASE,
)


def _event_uses_save_preference_tool(event: Event) -> bool:
    """Return True when an ADK ``Event`` invokes or responds to ``save_wording_preference``."""
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return False
    for part in parts:
        fn_call = getattr(part, "function_call", None)
        if fn_call is not None and getattr(fn_call, "name", None) == "save_wording_preference":
            return True
        fn_resp = getattr(part, "function_response", None)
        if fn_resp is not None and getattr(fn_resp, "name", None) == "save_wording_preference":
            return True
    return False


def _is_style_tweak_message(text_body: str) -> bool:
    """Determine whether a standalone user turn expresses a style preference tweak."""
    cleaned = text_body.strip()
    if len(cleaned) < 3:
        return False
    try:
        from thrilled_to_announce.tools.style_detection import (
            detect_human_or_linkedin_style,
        )

        detected = detect_human_or_linkedin_style(
            message=cleaned,
            has_previous_draft=True,
        )
        if detected.get("status") == "ok" and str(
            detected.get("detected_style", "")
        ).endswith("STYLE_TWEAK"):
            return True
    except Exception:
        pass
    return bool(_SESSION_TWEAK_CUE_RE.search(cleaned))


def _make_event_dedup_key(
    session_key: str,
    event: Event,
    text_body: str,
    occurrence: int,
) -> tuple[str, str]:
    raw_id = getattr(event, "id", None)
    if isinstance(raw_id, str) and raw_id.strip():
        return (session_key, raw_id.strip())
    digest = hashlib.blake2b(
        f"{occurrence}:{getattr(event, 'author', '')}:{text_body}".encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    return (session_key, f"digest_{digest}")


def _extract_sqlite_file_from_uri(uri: str, base_dir: Path) -> Path:
    """Resolve a SQLite filesystem path from ``sqlite:///`` or ``sqlite+aiosqlite:///``."""
    prefix_match = re.match(r"^sqlite(?:\+\w+)?:///(.*)$", uri, flags=re.IGNORECASE)
    if not prefix_match:
        return (base_dir / "adk_sessions.sqlite3").resolve()

    remainder = prefix_match.group(1)
    if not remainder or remainder == ":memory:":
        return (base_dir / "adk_sessions.sqlite3").resolve()

    if remainder.startswith("/"):
        return Path(remainder).resolve()

    top_segment = remainder.split("/", 1)[0]
    candidate_abs = Path("/" + remainder)
    if top_segment in {"tmp", "var", "usr", "home", "opt", "etc", "mnt", "run"} or (
        candidate_abs.parent.exists() and not (base_dir / remainder).parent.exists()
    ):
        return candidate_abs.resolve()

    return (base_dir / remainder).resolve()


def _normalize_session_uri(raw_uri: str, base_dir: Path) -> tuple[str, Path | None]:
    """Normalize ``session_service_uri`` for async SQLAlchemy or ``SqliteSessionService``."""
    cleaned = raw_uri.strip()
    lower = cleaned.lower()
    if lower.startswith("postgresql://"):
        return "postgresql+asyncpg://" + cleaned[len("postgresql://") :], None
    if lower.startswith("postgres://"):
        return "postgresql+asyncpg://" + cleaned[len("postgres://") :], None
    if lower.startswith("postgresql+"):
        return cleaned, None
    if lower.startswith("sqlite"):
        sqlite_file = _extract_sqlite_file_from_uri(cleaned, base_dir)
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        normalized_sqlalchemy_uri = f"sqlite+aiosqlite:///{sqlite_file}"
        return normalized_sqlalchemy_uri, sqlite_file
    return cleaned, None


def create_session_service(
    data_dir: Path | None = None,
    session_service_uri: str | None = None,
    use_vertex: bool = False,
    project_id: str | None = None,
    location: str = "us-central1",
    agent_engine_id: str | None = None,
) -> BaseSessionService:
    """Create a durable ADK ``BaseSessionService`` without ephemeral memory-only state.

    Selection order:
      1. ``VertexAiSessionService`` when ``agentengine://`` URI is supplied or
         ``use_vertex=True`` with a configured ``agent_engine_id``.
      2. ``DatabaseSessionService`` for ``postgresql+asyncpg://`` (Cloud SQL /
         AlloyDB) or ``sqlite:///`` / ``sqlite+aiosqlite:///`` URIs when
         SQLAlchemy is installed, falling back cleanly to ``SqliteSessionService``
         for SQLite URIs in lightweight environments.
      3. ``SqliteSessionService`` backed by ``${THRILLED_DATA_DIR}/adk_sessions.sqlite3``.
    """
    cfg = get_runtime_settings()
    resolved_dir = (Path(data_dir) if data_dir is not None else cfg.data_dir).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    effective_uri = (
        session_service_uri
        if session_service_uri is not None
        else (cfg.session_service_uri if data_dir is None else None)
    )
    effective_project = project_id if project_id is not None else cfg.google_cloud_project
    effective_location = location or cfg.google_cloud_location
    effective_engine_id = (
        agent_engine_id if agent_engine_id is not None else cfg.agent_engine_id
    )
    effective_use_vertex = bool(
        use_vertex or (cfg.google_genai_use_vertexai and effective_engine_id)
    )

    if effective_uri and effective_uri.strip().lower().startswith("agentengine://"):
        effective_engine_id = effective_uri.strip()[len("agentengine://") :].strip("/")
        effective_use_vertex = True

    if effective_use_vertex and effective_engine_id:
        try:
            return VertexAiSessionService(
                project=effective_project,
                location=effective_location,
                agent_engine_id=effective_engine_id,
            )
        except (ImportError, ValueError) as exc:
            get_structured_logger(__name__).warning(
                "vertex_session_service_fallback",
                reason=str(exc),
            )

    if effective_uri and not effective_uri.strip().lower().startswith("agentengine://"):
        normalized_uri, sqlite_file = _normalize_session_uri(
            effective_uri, resolved_dir
        )
        if DatabaseSessionService is not None:
            return DatabaseSessionService(db_url=normalized_uri)
        if sqlite_file is not None:
            sqlite_file.parent.mkdir(parents=True, exist_ok=True)
            return SqliteSessionService(db_path=str(sqlite_file.resolve()))
        raise RuntimeError(
            "DatabaseSessionService requires 'sqlalchemy' and 'asyncpg' to connect "
            f"to non-SQLite session_service_uri '{normalized_uri}'."
        )

    default_sqlite_file = (resolved_dir / "adk_sessions.sqlite3").resolve()
    default_sqlite_file.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSessionService(db_path=str(default_sqlite_file))


def _scrub_genai_content(content: types.Content | None) -> tuple[types.Content | None, str]:
    if content is None or not content.parts:
        return content, ""
    new_parts: list[types.Part] = []
    collected_texts: list[str] = []
    mutated = False
    for part in content.parts:
        if part.text is not None:
            scrubbed, redactions, _ = scrub_pii_text(part.text)
            collected_texts.append(scrubbed)
            if redactions > 0:
                mutated = True
                new_parts.append(types.Part.from_text(text=scrubbed))
            else:
                new_parts.append(part)
        else:
            new_parts.append(part)
    if not mutated:
        return content, "\n".join(collected_texts)
    return types.Content(role=content.role, parts=new_parts), "\n".join(collected_texts)


def _scrub_adk_event(event: Event) -> tuple[Event, str]:
    scrubbed_content, combined_text = _scrub_genai_content(event.content)
    if scrubbed_content is event.content:
        return event, combined_text
    return event.model_copy(update={"content": scrubbed_content}), combined_text


class PersistentVectorMemoryService(BaseMemoryService):
    """ADK ``BaseMemoryService`` combining ``VectorStyleMemoryStore`` and ``SqliteMemoryService``."""

    def __init__(
        self,
        style_store: VectorStyleMemoryStore | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        cfg = get_runtime_settings()
        resolved_dir = (
            Path(data_dir)
            if data_dir is not None
            else (style_store.data_dir if style_store is not None else cfg.data_dir)
        ).resolve()
        resolved_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir: Path = resolved_dir
        self.style_store: VectorStyleMemoryStore = style_store or get_default_style_store(
            data_dir=resolved_dir
        )
        self._events_memory = SqliteMemoryService(
            db_path=str((resolved_dir / "adk_events_memory.sqlite3").resolve())
        )
        self._consolidated_event_ids: set[tuple[str, str]] = set()

    def _collect_unseen_turn_events(
        self,
        *,
        session_key: str,
        events: Sequence[Event],
    ) -> tuple[list[Event], list[tuple[Event, str]]]:
        scrubbed_events: list[Event] = []
        unseen_events: list[tuple[Event, str]] = []
        fallback_counts: dict[str, int] = {}
        for event in events:
            scrubbed_ev, text_body = _scrub_adk_event(event)
            scrubbed_events.append(scrubbed_ev)
            sig_seed = f"{getattr(event, 'author', '')}:{text_body}"
            occ = fallback_counts.get(sig_seed, 0)
            fallback_counts[sig_seed] = occ + 1
            dedup_key = _make_event_dedup_key(session_key, event, text_body, occ)
            if dedup_key not in self._consolidated_event_ids:
                self._consolidated_event_ids.add(dedup_key)
                unseen_events.append((event, text_body))
        return scrubbed_events, unseen_events

    async def add_session_to_memory(self, session: Session) -> None:
        """Scrub PII from session events, persist to SQLite FTS, and consolidate style rules."""
        session_key = str(getattr(session, "id", None) or "default_session")
        scrubbed_events, unseen_events = self._collect_unseen_turn_events(
            session_key=session_key,
            events=session.events,
        )
        handled_by_save_tool = any(
            _event_uses_save_preference_tool(ev) for ev, _ in unseen_events
        )
        if not handled_by_save_tool:
            for event, text_body in unseen_events:
                if event.author == "user" and _is_style_tweak_message(text_body):
                    self.style_store.schedule_async_consolidation(
                        user_id=session.user_id,
                        raw_instruction=text_body.strip(),
                    )

        scrubbed_session = session.model_copy(update={"events": scrubbed_events})
        await self._events_memory.add_session_to_memory(scrubbed_session)
        await self.style_store.drain_background_tasks()

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Scrub PII from incremental ADK events and persist into durable memory."""
        session_key = str(session_id or f"{app_name}:{user_id}")
        scrubbed_events, unseen_events = self._collect_unseen_turn_events(
            session_key=session_key,
            events=events,
        )
        handled_by_save_tool = any(
            _event_uses_save_preference_tool(ev) for ev, _ in unseen_events
        )
        if not handled_by_save_tool:
            for event, text_body in unseen_events:
                if event.author == "user" and _is_style_tweak_message(text_body):
                    self.style_store.schedule_async_consolidation(
                        user_id=user_id,
                        raw_instruction=text_body.strip(),
                    )

        await self._events_memory.add_events_to_memory(
            app_name=app_name,
            user_id=user_id,
            events=scrubbed_events,
            session_id=session_id,
            custom_metadata=custom_metadata,
        )
        await self.style_store.drain_background_tasks()

    async def add_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        memories: Sequence[MemoryEntry],
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Scrub PII from explicit ``MemoryEntry`` items before SQLite storage."""
        scrubbed_memories: list[MemoryEntry] = []
        for entry in memories:
            scrubbed_content, _ = _scrub_genai_content(entry.content)
            if scrubbed_content is not None and scrubbed_content is not entry.content:
                scrubbed_memories.append(
                    entry.model_copy(update={"content": scrubbed_content})
                )
            else:
                scrubbed_memories.append(entry)
        await self._events_memory.add_memory(
            app_name=app_name,
            user_id=user_id,
            memories=scrubbed_memories,
            custom_metadata=custom_metadata,
        )

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        """Search user style preferences, dense vector idioms, and scrubbed session history."""
        scrubbed_query, _, _ = scrub_pii_text(query)
        now_iso = datetime.now(timezone.utc).isoformat()
        entries: list[MemoryEntry] = []

        profile = await self.style_store.get_profile_async(user_id=user_id)
        if profile.has_custom_preferences:
            banned_str = (
                ", ".join(profile.banned_words) if profile.banned_words else "none"
            )
            summary_text = (
                f"Saved wording style preference: tone='{profile.tone_summary}', "
                f"format='{profile.preferred_format.value}', "
                f"keep_concise={profile.keep_concise}, "
                f"avoid_emojis={profile.avoid_emojis}, "
                f"banned_words=[{banned_str}]."
            )
            entries.append(
                MemoryEntry(
                    id=f"profile_{user_id}",
                    author="style_memory",
                    timestamp=now_iso,
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=summary_text)],
                    ),
                    custom_metadata={"kind": "wording_preference_profile"},
                )
            )

        similar_examples = await self.style_store.search_similar_examples_async(
            user_id=user_id,
            query_text=scrubbed_query,
            top_k=3,
        )
        for ex in similar_examples:
            example_text = (
                f"Approved wording example (human -> LinkedIn): "
                f"'{ex.human_text}' -> '{ex.linkedin_text}'"
            )
            entries.append(
                MemoryEntry(
                    id=ex.entry_id,
                    author="style_memory",
                    timestamp=now_iso,
                    content=types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=example_text)],
                    ),
                    custom_metadata={
                        "kind": "idiom_vector_match",
                        "similarity_score": ex.similarity_score,
                    },
                )
            )

        if scrubbed_query.strip():
            fts_response = await self._events_memory.search_memory(
                app_name=app_name,
                user_id=user_id,
                query=scrubbed_query,
            )
            entries.extend(fts_response.memories)

        return SearchMemoryResponse(memories=entries)


def create_memory_service(
    data_dir: Path | None = None,
    use_vertex_memory_bank: bool = False,
    project_id: str | None = None,
    location: str = "us-central1",
    agent_engine_id: str | None = None,
    style_store: VectorStyleMemoryStore | None = None,
    memory_service_uri: str | None = None,
) -> BaseMemoryService:
    """Create ``VertexAiMemoryBankService`` (cloud) or ``PersistentVectorMemoryService``."""
    cfg = get_runtime_settings()
    resolved_dir = (
        Path(data_dir)
        if data_dir is not None
        else (style_store.data_dir if style_store is not None else cfg.data_dir)
    ).resolve()
    resolved_dir.mkdir(parents=True, exist_ok=True)

    effective_project = project_id if project_id is not None else cfg.google_cloud_project
    effective_location = location or cfg.google_cloud_location
    effective_engine_id = (
        agent_engine_id if agent_engine_id is not None else cfg.agent_engine_id
    )
    effective_use_bank = bool(use_vertex_memory_bank or cfg.use_vertex_memory_bank)

    if memory_service_uri and memory_service_uri.strip().lower().startswith(
        "agentengine://"
    ):
        effective_engine_id = memory_service_uri.strip()[
            len("agentengine://") :
        ].strip("/")
        effective_use_bank = True

    if effective_use_bank and effective_engine_id:
        try:
            return VertexAiMemoryBankService(
                project=effective_project,
                location=effective_location,
                agent_engine_id=effective_engine_id,
            )
        except (ImportError, ValueError) as exc:
            get_structured_logger(__name__).warning(
                "vertex_memory_bank_fallback",
                reason=str(exc),
            )

    return PersistentVectorMemoryService(
        style_store=style_store,
        data_dir=resolved_dir,
    )


async def _ingest_session_background(
    memory_service: BaseMemoryService,
    session: Session,
) -> None:
    try:
        await memory_service.add_session_to_memory(session)
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        get_structured_logger(__name__).warning(
            "after_agent_memory_ingestion_failed",
            error_message=scrubbed_err,
        )


def after_agent_memory_callback(callback_context: Any) -> None:
    """ADK ``after_agent_callback`` scheduling non-blocking ``add_session_to_memory``."""
    inv_ctx = getattr(callback_context, "_invocation_context", None)
    session: Session | None = getattr(callback_context, "session", None) or getattr(
        inv_ctx, "session", None
    )
    if session is None:
        return None

    memory_service: BaseMemoryService | None = getattr(
        callback_context, "memory_service", None
    ) or getattr(inv_ctx, "memory_service", None)
    if memory_service is None:
        memory_service = create_memory_service()

    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    task = running_loop.create_task(
        _ingest_session_background(memory_service, session)
    )
    _BACKGROUND_MEMORY_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_MEMORY_TASKS.discard)
    return None


after_agent_callback = after_agent_memory_callback


async def drain_memory_tasks() -> None:
    """Await all queued ``after_agent_memory_callback`` and style consolidation tasks."""
    while _BACKGROUND_MEMORY_TASKS:
        pending = list(_BACKGROUND_MEMORY_TASKS)
        results = await asyncio.gather(*pending, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                scrubbed_err, _, _ = scrub_pii_text(str(item))
                get_structured_logger(__name__).warning(
                    "background_memory_task_failed",
                    error_message=scrubbed_err,
                )
    await drain_all_style_tasks()


__all__ = [
    "DatabaseSessionService",
    "MemoryEntry",
    "PersistentVectorMemoryService",
    "PreloadMemoryTool",
    "SearchMemoryResponse",
    "SqliteMemoryService",
    "SqliteSessionService",
    "VertexAiMemoryBankService",
    "VertexAiSessionService",
    "after_agent_callback",
    "after_agent_memory_callback",
    "create_memory_service",
    "create_session_service",
    "drain_memory_tasks",
    "load_memory",
    "preload_memory",
    "preload_memory_tool",
]
