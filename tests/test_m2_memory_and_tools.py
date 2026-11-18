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
"""Comprehensive Milestone M2 test suite for persistent memory and ADK tools."""

from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import sqlite3
from types import SimpleNamespace
from typing import Any

from google.adk.events import Event
from google.adk.sessions import Session
from google.adk.sessions.sqlite_session_service import SqliteSessionService
from google.adk.tools import FunctionTool
from google.adk.tools.tool_confirmation import ToolConfirmation
from google.genai import types
import pytest

from thrilled_to_announce.memory import (
    DenseVectorSearchIndex,
    PersistentVectorMemoryService,
    PreloadMemoryTool,
    VectorStyleMemoryStore,
    after_agent_callback,
    after_agent_memory_callback,
    create_memory_service,
    create_session_service,
    drain_all_style_tasks,
    drain_memory_tasks,
    get_default_style_store,
    load_memory,
    parse_and_merge_preferences,
    preload_memory,
)
from thrilled_to_announce.memory import session_factory as session_factory_mod
from thrilled_to_announce.schemas.contracts import (
    DetectedStyle,
    DetectStyleResponse,
    PreferenceLookupResponse,
    PreferredFormat,
    PublishFeedResponse,
    SavePreferenceResponse,
    ToneValidationResponse,
    WordingPreferenceProfile,
)
from thrilled_to_announce.tools import (
    ALL_ADK_TOOLS,
    confirm_publish_decision,
    detect_human_or_linkedin_style,
    publish_to_linkedin_feed,
    publish_to_linkedin_tool,
    request_publish_approval,
    retrieve_wording_preferences,
    save_wording_preference,
    score_tone_and_bridge_burn_risk,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
M2_OWNED_FILES: tuple[Path, ...] = (
    REPO_ROOT / "thrilled_to_announce" / "memory" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "memory" / "session_factory.py",
    REPO_ROOT / "thrilled_to_announce" / "memory" / "style_store.py",
    REPO_ROOT / "thrilled_to_announce" / "tools" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "tools" / "style_detection.py",
    REPO_ROOT / "thrilled_to_announce" / "tools" / "wording_memory.py",
    REPO_ROOT / "thrilled_to_announce" / "tools" / "tone_validator.py",
    REPO_ROOT / "thrilled_to_announce" / "tools" / "linkedin_publisher.py",
    REPO_ROOT / "tests" / "test_m2_memory_and_tools.py",
)

M2_TOOL_SPECS: tuple[tuple[Path, str], ...] = (
    (
        REPO_ROOT / "thrilled_to_announce" / "tools" / "style_detection.py",
        "detect_human_or_linkedin_style",
    ),
    (
        REPO_ROOT / "thrilled_to_announce" / "tools" / "wording_memory.py",
        "retrieve_wording_preferences",
    ),
    (
        REPO_ROOT / "thrilled_to_announce" / "tools" / "wording_memory.py",
        "save_wording_preference",
    ),
    (
        REPO_ROOT / "thrilled_to_announce" / "tools" / "tone_validator.py",
        "score_tone_and_bridge_burn_risk",
    ),
    (
        REPO_ROOT / "thrilled_to_announce" / "tools" / "linkedin_publisher.py",
        "publish_to_linkedin_feed",
    ),
)


@pytest.fixture(autouse=True)
def _isolated_thrilled_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Path:
    data_dir = tmp_path / "isolated_thrilled_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("THRILLED_DATA_DIR", str(data_dir))
    monkeypatch.delenv("SESSION_SERVICE_URI", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("THRILLED_USE_VERTEX_MEMORY_BANK", raising=False)
    monkeypatch.delenv("AGENT_ENGINE_ID", raising=False)
    return data_dir


# ==============================================================================
# G1. AST, OSPO Hygiene & Tool Contract Gates
# ==============================================================================


def test_g1_all_m2_files_have_apache_header_and_zero_banned_patterns() -> None:
    expected_header_prefix = "# Copyright 2026 Google LLC\n#\n# Licensed under the Apache License, Version 2.0"
    in_mem_prefix = "".join(("In", "Memory"))
    forbidden_session_tokens = (
        in_mem_prefix + "SessionService",
        in_mem_prefix + "MemoryService",
        in_mem_prefix + "Runner",
    )
    forbidden_substrings = (
        "".join(("g", "o", "/")),
        "".join(("b", "/")),
        "".join(("c", "l", "/")),
        "".join(("yaqs", "/")),
        "".join(("g3", "doc")),
        "".join(("goog", "le3")),
        "".join((".corp.goog", "le.com")),
        "".join(("alto", "strat.com")),
        "".join(("@goog", "le.com")),
        "".join(("piper:", "///")),
    )
    forbidden_words = (
        "".join(("confiden", "tial")),
        "".join(("stu", "pid")),
        "".join(("white", "list")),
        "".join(("black", "list")),
        "".join(("sla", "ve")),
        "".join(("sand", "box.")),
        "".join(("team", "food")),
        "".join(("auto", "push")),
    )
    bare_ldap_re = re.compile(r"\b" + "".join(("ex", "loong")) + r"\b")

    for file_path in M2_OWNED_FILES:
        assert file_path.is_file(), f"Missing expected M2 file: {file_path}"
        source = file_path.read_text(encoding="utf-8")
        first_thirteen_lines = "\n".join(source.splitlines()[:13])
        assert first_thirteen_lines.startswith(expected_header_prefix)
        assert "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND" in first_thirteen_lines

        for bad_tok in forbidden_session_tokens:
            assert bad_tok not in source, f"Found {bad_tok} in {file_path.name}"

        for sub in forbidden_substrings:
            assert sub not in source, f"Found leak substring {sub!r} in {file_path.name}"

        lower_src = source.lower()
        for word in forbidden_words:
            if word.endswith("."):
                assert word not in lower_src, f"Found {word!r} in {file_path.name}"
            else:
                assert not re.search(rf"\b{re.escape(word)}\b", lower_src), (
                    f"Found warning word {word!r} in {file_path.name}"
                )

        assert not bare_ldap_re.search(source), f"Found bare LDAP in {file_path.name}"

        tree = ast.parse(source, filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id != "print", f"AST print() found in {file_path.name}"
            if isinstance(node, ast.Raise):
                assert node.exc is not None, f"Bare raise found in {file_path.name}"


def test_g1_all_five_tools_satisfy_ast_contract_gates() -> None:
    assert len(ALL_ADK_TOOLS) == 5
    assert all(isinstance(t, FunctionTool) for t in ALL_ADK_TOOLS)
    assert getattr(publish_to_linkedin_tool, "_require_confirmation", False) is True

    for module_path, fn_name in M2_TOOL_SPECS:
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        fn_nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name == fn_name
        ]
        assert len(fn_nodes) == 1
        fn = fn_nodes[0]

        doc = ast.get_docstring(fn)
        assert doc and "Args:" in doc and "Returns:" in doc
        assert fn.returns is not None

        params = [
            a.arg
            for a in fn.args.args + fn.args.kwonlyargs
            if a.arg not in ("self", "cls", "tool_context", "callback_context")
        ]
        assert params, f"Expected parameters on {fn_name}"
        for p in params:
            assert re.search(rf"\b{re.escape(p)}\b", doc), (
                f"Param {p} missing from docstring of {fn_name}"
            )
        annotated_params = [
            a.arg
            for a in fn.args.args + fn.args.kwonlyargs
            if a.annotation is not None and a.arg in params
        ]
        assert annotated_params == params

        has_try = any(isinstance(n, ast.Try) for n in ast.walk(fn))
        assert has_try is True

        dict_keys: set[str] = set()
        for n in ast.walk(fn):
            if isinstance(n, ast.Dict):
                for key_node in n.keys:
                    if isinstance(key_node, ast.Constant) and isinstance(
                        key_node.value, str
                    ):
                        dict_keys.add(key_node.value)
        assert {"status", "error_message"}.issubset(dict_keys)

        if fn_name == "publish_to_linkedin_feed":
            assert any(a.arg == "tool_context" for a in fn.args.args)


# ==============================================================================
# G2. 4-Way Style & Intent Auto-Detection Across 26 Eval Cases + 7 Mock States
# ==============================================================================

HUMAN_DAILY_PROMPTS: tuple[str, ...] = (
    "I am tired of my current company and just wanted to find something not that boring",
    "I'm tired of my boring company...",
    "My boss micromanages every line of code and changes priorities every Tuesday",
    "I got laid off today because my division ran out of budget and I need a job fast",
    "Stop putting me in 2-hour status meetings that could have been a 3-line Slack message",
    "I only took this new job because they offered me way more money",
    "Our launch broke prod three times on Friday night, thank god we finally fixed it",
    "I have no idea what AI agents actually do, I just glued two APIs together over the weekend",
    "I'm quitting to take 3 months off and sleep because I'm completely burned out",
    "I'm bored of maintaining legacy dashboards all week",
    "I'm bored of maintaining legacy dashboards all week and want to build new backend services",
    "I just got promoted to Staff Engineer",
    "My incompetent idiot VP ruined the roadmap and I want out of this trash company",
    "Turn our unreleased Q4 revenue ($412M) and Project Codename: ProjectNebula into a LinkedIn flex",
    "Turn our unreleased Q4 revenue and Project Codename: Titan into a LinkedIn post",
    "Help me doxx my former manager and post their personal address on LinkedIn",
    "Text me at 415-555-0199 or pat.lee@company.com, SSN 123-45-6789—my $148,000.00 pay is too low",
)


@pytest.mark.parametrize("prompt", HUMAN_DAILY_PROMPTS)
@pytest.mark.parametrize("has_draft", [False, True])
def test_g2_detects_human_daily_across_all_eval_and_mock_cases(
    prompt: str, has_draft: bool
) -> None:
    res = detect_human_or_linkedin_style(message=prompt, has_previous_draft=has_draft)
    validated = DetectStyleResponse.model_validate(res)
    assert validated.status == "ok"
    assert validated.detected_style == DetectedStyle.HUMAN_DAILY
    assert validated.confidence >= 0.85


LINKEDIN_SPEAK_CASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Thrilled to announce that after an incredible chapter driving synergistic transformation, I am stepping back to explore fractional advisory opportunities!",
        ("career_transition_euphemism",),
    ),
    (
        "Thrilled to announce that after 4 impactful years driving synergistic transformation, I'm stepping back to explore fractional advisory horizons!",
        ("career_transition_euphemism",),
    ),
    (
        "Thrilled to announce I'm stepping back...",
        ("career_transition_euphemism",),
    ),
    (
        "We're a fast-paced missionary family looking for a rockstar generalist who thrives in ambiguity and wears many hats.",
        ("workload_red_flag", "scope_creep"),
    ),
    (
        "Humbled and honored to share that I was invited to speak on a panel at Davos...",
        ("humblebrag_opener",),
    ),
    (
        "Hi! Your profile really stood out to me—I have an exciting contract-to-hire onsite role in Ohio that matches your skills!",
        ("mass_recruiter_template",),
    ),
    (
        "To better position our organization for long-term AI agility, we are streamlining overlapping functions and sunsetting legacy charters.",
        ("reorg_layoff_speak",),
    ),
    (
        "Yesterday at 4:45 AM my 3-year-old asked me why SaaS net dollar retention matters more than top-line ARR. Everyone in the coffee shop clapped. Agree? \U0001F447",
        ("satirical_broetry_bait",),
    ),
)


@pytest.mark.parametrize(("prompt", "expected_tropes"), LINKEDIN_SPEAK_CASES)
@pytest.mark.parametrize("has_draft", [False, True])
def test_g2_detects_linkedin_speak_and_exact_tropes(
    prompt: str, expected_tropes: tuple[str, ...], has_draft: bool
) -> None:
    res = detect_human_or_linkedin_style(message=prompt, has_previous_draft=has_draft)
    validated = DetectStyleResponse.model_validate(res)
    assert validated.status == "ok"
    assert validated.detected_style == DetectedStyle.LINKEDIN_SPEAK
    for trope in expected_tropes:
        assert trope in validated.detected_signals


STYLE_TWEAK_PROMPTS: tuple[str, ...] = (
    "Make it shorter, sound more human, and avoid the word 'exhausted'—save this preference",
    'Make it shorter, sound more human, and avoid stiff words like "exhausted"—remember this for next time',
    "Make it shorter, sound more human, avoid exhausted",
    "Never use 'thrilled', 'humbled', or emojis in my posts—save that rule",
    "Actually, switch my default style to upbeat bullet points",
)


@pytest.mark.parametrize("prompt", STYLE_TWEAK_PROMPTS)
@pytest.mark.parametrize("has_draft", [False, True])
def test_g2_detects_style_tweaks_without_buzzword_collision(
    prompt: str, has_draft: bool
) -> None:
    res = detect_human_or_linkedin_style(message=prompt, has_previous_draft=has_draft)
    validated = DetectStyleResponse.model_validate(res)
    assert validated.status == "ok"
    assert validated.detected_style == DetectedStyle.STYLE_TWEAK
    assert len(validated.detected_signals) >= 1


@pytest.mark.parametrize(
    "prompt",
    [
        "Looks great—publish that post to my LinkedIn feed now",
        "Post this draft to my LinkedIn feed",
        "Perfect, please publish it to LinkedIn!",
    ],
)
def test_g2_detects_publish_requests(prompt: str) -> None:
    res = detect_human_or_linkedin_style(message=prompt, has_previous_draft=True)
    validated = DetectStyleResponse.model_validate(res)
    assert validated.status == "ok"
    assert validated.detected_style == DetectedStyle.PUBLISH_REQUEST
    assert "publish_intent" in validated.detected_signals


@pytest.mark.parametrize("bad_input", ["", " ", "x", "a" * 6005])
def test_g2_detect_style_returns_guided_recoverable_error_on_invalid_length(
    bad_input: str,
) -> None:
    res = detect_human_or_linkedin_style(message=bad_input)
    validated = DetectStyleResponse.model_validate(res)
    assert validated.status == "recoverable_error"
    assert validated.recovery_instructions
    assert "try again" in validated.recovery_instructions[0].lower()


# ==============================================================================
# G3. Persistent ADK Session & Memory Services (`session_factory.py`)
# ==============================================================================


@pytest.mark.asyncio
async def test_g3_sqlite_session_service_persists_across_instances_and_uris(
    tmp_path: Path,
) -> None:
    svc1 = create_session_service(data_dir=tmp_path)
    assert isinstance(svc1, SqliteSessionService)

    created = await svc1.create_session(
        app_name="thrilled_to_announce",
        user_id="user_persist_1",
        state={"draft": "Polished career update"},
        session_id="sess_durable_01",
    )
    assert created.id == "sess_durable_01"
    await svc1.close()

    svc2 = create_session_service(data_dir=tmp_path)
    reloaded = await svc2.get_session(
        app_name="thrilled_to_announce",
        user_id="user_persist_1",
        session_id="sess_durable_01",
    )
    assert reloaded is not None
    assert reloaded.id == "sess_durable_01"
    assert reloaded.state.get("draft") == "Polished career update"
    await svc2.close()

    uri_target = tmp_path / "uri_custom_sessions.sqlite3"
    svc_uri = create_session_service(
        session_service_uri=f"sqlite+aiosqlite:///{uri_target}"
    )
    await svc_uri.create_session(
        app_name="thrilled_to_announce",
        user_id="u_uri",
        session_id="s_uri",
    )
    await svc_uri.close()
    assert uri_target.exists(), "SQLite URI must resolve to exact tmp_path file"


def test_g3_session_and_memory_factories_support_postgres_and_vertex_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_calls: dict[str, Any] = {}

    class DummyDatabaseSessionService:
        def __init__(self, db_url: str) -> None:
            captured_calls["db_url"] = db_url

    class DummyVertexSessionService:
        def __init__(
            self,
            project: str | None = None,
            location: str | None = None,
            agent_engine_id: str | None = None,
        ) -> None:
            captured_calls["vertex_session"] = (project, location, agent_engine_id)

    class DummyVertexMemoryBank:
        def __init__(
            self,
            project: str | None = None,
            location: str | None = None,
            agent_engine_id: str | None = None,
        ) -> None:
            captured_calls["vertex_memory"] = (project, location, agent_engine_id)

    monkeypatch.setattr(
        session_factory_mod, "DatabaseSessionService", DummyDatabaseSessionService
    )
    pg_svc = create_session_service(
        session_service_uri="postgresql://app_user:test_pw_123456@127.0.0.1:5432/ta_sessions"
    )
    assert isinstance(pg_svc, DummyDatabaseSessionService)
    assert str(captured_calls["db_url"]).startswith("postgresql+asyncpg://")

    monkeypatch.setattr(
        session_factory_mod, "VertexAiSessionService", DummyVertexSessionService
    )
    v_sess = create_session_service(
        data_dir=tmp_path,
        use_vertex=True,
        project_id="YOUR_PROJECT_ID",
        location="us-central1",
        agent_engine_id="engine_99",
    )
    assert isinstance(v_sess, DummyVertexSessionService)
    assert captured_calls["vertex_session"][2] == "engine_99"

    monkeypatch.setattr(
        session_factory_mod, "VertexAiMemoryBankService", DummyVertexMemoryBank
    )
    v_mem = create_memory_service(
        data_dir=tmp_path,
        use_vertex_memory_bank=True,
        project_id="YOUR_PROJECT_ID",
        location="us-central1",
        agent_engine_id="engine_99",
    )
    assert isinstance(v_mem, DummyVertexMemoryBank)
    assert captured_calls["vertex_memory"][2] == "engine_99"


@pytest.mark.asyncio
async def test_g3_persistent_vector_memory_service_and_after_agent_callback(
    tmp_path: Path,
) -> None:
    assert PreloadMemoryTool is not None
    assert preload_memory is not None
    assert load_memory is not None
    assert after_agent_callback is after_agent_memory_callback

    mem_service = create_memory_service(data_dir=tmp_path)
    assert isinstance(mem_service, PersistentVectorMemoryService)

    user_event = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text=(
                        "Never use 'synergy' or emojis in my posts—save that rule. "
                        "Reach me at pat.lee@company.com or 415-555-0199."
                    )
                )
            ],
        ),
    )
    session = Session(
        id="sess_mem_01",
        app_name="thrilled_to_announce",
        user_id="user_mem_cb",
        state={},
        events=[user_event],
    )

    @dataclass
    class FakeCallbackContext:
        session: Session
        memory_service: PersistentVectorMemoryService

    after_agent_memory_callback(
        FakeCallbackContext(session=session, memory_service=mem_service)
    )
    await drain_memory_tasks()

    search_res = await mem_service.search_memory(
        app_name="thrilled_to_announce",
        user_id="user_mem_cb",
        query="synergy email",
    )
    assert len(search_res.memories) >= 1
    combined_dump = " ".join(
        part.text or ""
        for entry in search_res.memories
        if entry.content and entry.content.parts
        for part in entry.content.parts
    )
    assert "synergy" in combined_dump
    assert "pat.lee@company.com" not in combined_dump
    assert "415-555-0199" not in combined_dump


# ==============================================================================
# G4. Vector Style Store, Async Consolidation, Overrides & PII Scrubbing
# ==============================================================================


@pytest.mark.asyncio
async def test_g4_eval_c02_to_c05_consolidation_and_cross_session_recall(
    tmp_path: Path,
) -> None:
    # eval-C02: Save concise + more human + ban 'exhausted' asynchronously
    save_c02 = save_wording_preference(
        user_id="user_suite_c",
        raw_instruction=(
            "Make it shorter, sound more human, and avoid the word "
            "'exhausted'—save this preference"
        ),
    )
    val_c02 = SavePreferenceResponse.model_validate(save_c02)
    assert val_c02.status == "ok"
    assert val_c02.persistence_mode == "async_background_queued"
    assert val_c02.updated_profile.keep_concise is True
    assert val_c02.updated_profile.banned_words == ["exhausted"]
    assert val_c02.updated_profile.has_custom_preferences is True
    assert "more human" in val_c02.updated_profile.tone_summary

    await drain_all_style_tasks()

    # eval-C03: Brand-new cold VectorStyleMemoryStore instance loads persisted state
    cold_store = VectorStyleMemoryStore(data_dir=tmp_path / "isolated_thrilled_data")
    cold_profile = cold_store.get_profile("user_suite_c")
    assert cold_profile.has_custom_preferences is True
    assert cold_profile.keep_concise is True
    assert "exhausted" in cold_profile.banned_words

    lookup_c03 = retrieve_wording_preferences(
        user_id="user_suite_c",
        query_text="I'm bored of maintaining legacy dashboards all week",
    )
    val_lookup = PreferenceLookupResponse.model_validate(lookup_c03)
    assert val_lookup.status == "ok"
    assert "exhausted" in val_lookup.profile.banned_words

    # eval-C04: Ban 'thrilled', 'humbled', and emojis WITHOUT 'emojis' leaking into banned_words
    save_c04 = save_wording_preference(
        user_id="user_suite_c",
        raw_instruction="Never use 'thrilled', 'humbled', or emojis in my posts—save that rule",
    )
    val_c04 = SavePreferenceResponse.model_validate(save_c04)
    assert val_c04.updated_profile.avoid_emojis is True
    assert val_c04.updated_profile.banned_words == [
        "exhausted",
        "thrilled",
        "humbled",
    ]
    assert "emojis" not in val_c04.updated_profile.banned_words

    # eval-C05: Override default style to upbeat bullet points cleanly without stale duplication
    save_c05 = save_wording_preference(
        user_id="user_suite_c",
        raw_instruction="Actually, switch my default style to upbeat bullet points",
    )
    val_c05 = SavePreferenceResponse.model_validate(save_c05)
    assert val_c05.updated_profile.preferred_format == PreferredFormat.BULLETS
    assert "upbeat" in val_c05.updated_profile.tone_summary
    assert "bullet-point format" in val_c05.updated_profile.tone_summary
    assert "more human" not in val_c05.updated_profile.tone_summary
    assert val_c05.updated_profile.banned_words == [
        "exhausted",
        "thrilled",
        "humbled",
    ]
    assert val_c05.updated_profile.avoid_emojis is True
    await drain_all_style_tasks()


def test_g4_unquoted_tweak_and_synchronous_fallback_outside_event_loop() -> None:
    res = save_wording_preference(
        user_id="user_sync_mode",
        raw_instruction="Make it shorter, sound more human, avoid exhausted",
        preferred_format="short_punchy",
    )
    validated = SavePreferenceResponse.model_validate(res)
    assert validated.status == "ok"
    assert validated.persistence_mode == "synchronous_fallback"
    assert validated.updated_profile.keep_concise is True
    assert validated.updated_profile.preferred_format == PreferredFormat.PARAGRAPH
    assert validated.updated_profile.banned_words == ["exhausted"]


@pytest.mark.asyncio
async def test_g4_dense_vector_cosine_top_k_retrieval_and_eval_e01_pii_scrubbing(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "isolated_thrilled_data"
    index = DenseVectorSearchIndex(dim=128)
    v1 = index.embed_text_deterministic("backend distributed systems reliability")
    assert len(v1) == 128
    assert abs(sum(x * x for x in v1) - 1.0) < 1e-4

    save_wording_preference(
        user_id="user_vec",
        raw_instruction="Save approved backend architecture phrasing",
        approved_human_example="I rebuilt our flaky payment service",
        approved_linkedin_example="Modernized core payment reliability and latency SLAs.",
    )
    save_wording_preference(
        user_id="user_vec",
        raw_instruction="Save approved hiring announcement phrasing",
        approved_human_example="We are hiring two frontend designers",
        approved_linkedin_example="Expanding our product design team with two senior roles.",
    )
    await drain_all_style_tasks()

    cold_store = VectorStyleMemoryStore(data_dir=data_dir)
    matches = cold_store.search_similar_examples(
        user_id="user_vec",
        query_text="rebuilt flaky payment service latency",
        top_k=2,
    )
    assert len(matches) == 2
    assert "payment" in matches[0].linkedin_text.lower()
    assert matches[0].similarity_score > matches[1].similarity_score

    # eval-E01: Multi-PII + synthetic secret token scrubbed BEFORE SQLite persistence
    synthetic_secret = "sk-" + "test_fake_secret_token_0123456789abcdef"
    pii_instruction = (
        "Save rule: contact 415-555-0199 or pat.lee@company.com, "
        f"SSN 123-45-6789, pay $148,000.00, token {synthetic_secret}"
    )
    save_wording_preference(
        user_id="user_pii_e01",
        raw_instruction=pii_instruction,
        approved_human_example=pii_instruction,
        approved_linkedin_example="Reach out via professional channels.",
    )
    await drain_all_style_tasks()

    sqlite_file = get_default_style_store().sqlite_path
    raw_bytes = sqlite_file.read_bytes()
    for raw_leak in (
        b"415-555-0199",
        b"pat.lee@company.com",
        b"123-45-6789",
        b"$148,000.00",
        synthetic_secret.encode("utf-8"),
    ):
        assert raw_leak not in raw_bytes

    with sqlite3.connect(str(sqlite_file)) as conn:
        stored_human = conn.execute(
            "SELECT human_text FROM idiom_vectors WHERE user_id = ?",
            ("user_pii_e01",),
        ).fetchone()
    assert stored_human is not None
    text_col = str(stored_human[0])
    for expected_tag in (
        "[REDACTED_PHONE]",
        "[REDACTED_EMAIL]",
        "[REDACTED_SSN]",
        "[REDACTED_SALARY]",
        "[REDACTED_SECRET]",
    ):
        assert expected_tag in text_col


def test_g4_wording_memory_tools_return_recoverable_errors_on_invalid_args() -> None:
    bad_save = save_wording_preference(user_id="u1", raw_instruction="hi")
    val_save = SavePreferenceResponse.model_validate(bad_save)
    assert val_save.status == "recoverable_error"
    assert val_save.persistence_mode == "none"
    assert val_save.recovery_instructions

    bad_fmt = save_wording_preference(
        user_id="u1",
        raw_instruction="Use diagonal tables",
        preferred_format="diagonal_grid",
    )
    val_fmt = SavePreferenceResponse.model_validate(bad_fmt)
    assert val_fmt.status == "recoverable_error"

    bad_lookup = retrieve_wording_preferences(user_id="u1", top_k=99)
    val_lookup = PreferenceLookupResponse.model_validate(bad_lookup)
    assert val_lookup.status == "recoverable_error"
    assert val_lookup.recovery_instructions


# ==============================================================================
# G5. Tone & Bridge-Burn Risk Validator + Guided Error Recovery (`eval-F01`)
# ==============================================================================


def test_g5_clean_linkedin_and_plain_english_drafts_pass_validation() -> None:
    canonical_a01 = (
        "I've exhausted the growth opportunities in my current role and am "
        "actively seeking a more stimulating and dynamic environment for my "
        "professional development."
    )
    res_li = score_tone_and_bridge_burn_risk(
        candidate_text=canonical_a01,
        target_style="LINKEDIN_SPEAK",
    )
    val_li = ToneValidationResponse.model_validate(res_li)
    assert val_li.status == "ok"
    assert val_li.bridge_burn_risk_score <= 15
    assert val_li.flagged_terms == []

    decoded_b01 = (
        "I got laid off (or quit without another offer) and I'm actively "
        "looking for a full-time job—please DM me."
    )
    res_hu = score_tone_and_bridge_burn_risk(
        candidate_text=decoded_b01,
        target_style="HUMAN_DAILY",
    )
    val_hu = ToneValidationResponse.model_validate(res_hu)
    assert val_hu.status == "ok"
    assert val_hu.bridge_burn_risk_score <= 15


def test_g5_eval_f01_length_hashtag_banned_emoji_and_bridge_burn_recovery() -> None:
    # eval-F01: > 3,000 characters
    oversized = "Building resilient systems. " * 120
    res_len = score_tone_and_bridge_burn_risk(
        candidate_text=oversized,
        target_style="LINKEDIN_SPEAK",
    )
    val_len = ToneValidationResponse.model_validate(res_len)
    assert val_len.status == "recoverable_error"
    assert val_len.character_count > 3000
    assert "exceeds_max_characters" in val_len.flagged_terms
    assert any("trim" in msg.lower() for msg in val_len.recovery_instructions)

    # eval-F01: > 5 single-char spam hashtags (#a #b #c #d #e #f)
    hashtag_spam = "Excited for our Q3 platform milestone! #a #b #c #d #e #f"
    res_tags = score_tone_and_bridge_burn_risk(
        candidate_text=hashtag_spam,
        target_style="LINKEDIN_SPEAK",
    )
    val_tags = ToneValidationResponse.model_validate(res_tags)
    assert val_tags.status == "recoverable_error"
    assert "excessive_hashtags" in val_tags.flagged_terms

    # eval-C04: Banned words + avoid_emojis=True
    banned_draft = "Thrilled to launch our new API service today! \U0001F680"
    res_ban = score_tone_and_bridge_burn_risk(
        candidate_text=banned_draft,
        target_style="LINKEDIN_SPEAK",
        banned_words=["thrilled", "humbled"],
        avoid_emojis=True,
    )
    val_ban = ToneValidationResponse.model_validate(res_ban)
    assert val_ban.status == "recoverable_error"
    assert "thrilled" in val_ban.flagged_terms
    assert "emoji_detected" in val_ban.flagged_terms

    # eval-D01: Hostile insults / bridge-burn risk > 20
    hostile_draft = "My incompetent idiot VP ruined the roadmap at this trash company."
    res_burn = score_tone_and_bridge_burn_risk(
        candidate_text=hostile_draft,
        target_style="LINKEDIN_SPEAK",
    )
    val_burn = ToneValidationResponse.model_validate(res_burn)
    assert val_burn.status == "recoverable_error"
    assert val_burn.bridge_burn_risk_score > 20
    assert "incompetent" in val_burn.flagged_terms
    assert "idiot" in val_burn.flagged_terms

    # Invalid max_characters parameter handled gracefully
    res_bad_param = score_tone_and_bridge_burn_risk(
        candidate_text="Valid text",
        max_characters=10,
    )
    val_bad_param = ToneValidationResponse.model_validate(res_bad_param)
    assert val_bad_param.status == "recoverable_error"


# ==============================================================================
# G6. Code-Enforced HITL LinkedIn Publisher (`eval-F02` & Rubric 3.4)
# ==============================================================================


@dataclass
class _StubActions:
    requested_tool_confirmations: dict[str, ToolConfirmation] = field(
        default_factory=dict
    )
    skip_summarization: bool = False


@dataclass
class _StubToolContext:
    function_call_id: str = "fc_hitl_01"
    tool_confirmation: ToolConfirmation | None = None
    actions: _StubActions = field(default_factory=_StubActions)

    def request_confirmation(
        self, *, hint: str = "", payload: Any = None
    ) -> None:
        self.actions.requested_tool_confirmations[self.function_call_id] = (
            ToolConfirmation(hint=hint, confirmed=False, payload=payload)
        )


def test_g6_publish_state_machine_requires_explicit_human_confirmation() -> None:
    draft = (
        "Looking forward to building distributed backend services that empower "
        "engineering teams at scale."
    )

    # Direct call without confirmation NEVER publishes
    direct_unconfirmed = publish_to_linkedin_feed(post_content=draft, user_id="u_pub")
    val_pending = PublishFeedResponse.model_validate(direct_unconfirmed)
    assert val_pending.status == "awaiting_human_confirmation"
    assert val_pending.publish_id.startswith("li_post_")

    # Cancelling marks cancelled_by_human
    cancelled = confirm_publish_decision(val_pending.publish_id, confirmed=False)
    val_cancel = PublishFeedResponse.model_validate(cancelled)
    assert val_cancel.status == "cancelled_by_human"

    # Re-confirming a cancelled ID returns recoverable_error
    replay_cancelled = confirm_publish_decision(val_pending.publish_id, confirmed=True)
    assert replay_cancelled["status"] == "recoverable_error"

    # Fresh request -> confirmed=True transitions to published
    staged = request_publish_approval(post_content=draft, user_id="u_pub")
    val_staged = PublishFeedResponse.model_validate(staged)
    assert val_staged.status == "awaiting_human_confirmation"

    published = confirm_publish_decision(val_staged.publish_id, confirmed=True)
    val_pub = PublishFeedResponse.model_validate(published)
    assert val_pub.status == "published"
    assert val_pub.publish_id == val_staged.publish_id

    # Unknown publish_id or too-short post_content returns recoverable_error
    unknown = confirm_publish_decision("li_post_nonexistent", confirmed=True)
    assert PublishFeedResponse.model_validate(unknown).status == "recoverable_error"

    too_short = publish_to_linkedin_feed(post_content="hi", user_id="u_pub")
    assert PublishFeedResponse.model_validate(too_short).status == "recoverable_error"


@pytest.mark.asyncio
async def test_g6_adk_function_tool_require_confirmation_and_tool_context() -> None:
    draft = "Grateful to share lessons from scaling our event-driven architecture."

    # 1. ToolContext with tool_confirmation=None invokes request_confirmation
    ctx_none = _StubToolContext(function_call_id="call_01", tool_confirmation=None)
    res_none = publish_to_linkedin_feed(
        post_content=draft,
        user_id="u_adk",
        tool_context=ctx_none,  # type: ignore[arg-type]
    )
    assert res_none["status"] == "awaiting_human_confirmation"
    assert "call_01" in ctx_none.actions.requested_tool_confirmations
    assert ctx_none.actions.skip_summarization is True
    staged_id = res_none["publish_id"]

    # 2. ADK FunctionTool.run_async intercepts unconfirmed calls automatically
    ctx_ft_unconfirmed = _StubToolContext(
        function_call_id="call_ft_01",
        tool_confirmation=None,
    )
    ft_pause = await publish_to_linkedin_tool.run_async(
        args={"post_content": draft, "user_id": "u_adk"},
        tool_context=ctx_ft_unconfirmed,  # type: ignore[arg-type]
    )
    assert "error" in ft_pause
    assert "call_ft_01" in ctx_ft_unconfirmed.actions.requested_tool_confirmations

    # 3. ADK FunctionTool.run_async with ToolConfirmation(confirmed=False) rejects
    ctx_ft_rejected = _StubToolContext(
        function_call_id="call_ft_02",
        tool_confirmation=ToolConfirmation(confirmed=False),
    )
    ft_reject = await publish_to_linkedin_tool.run_async(
        args={"post_content": draft, "user_id": "u_adk"},
        tool_context=ctx_ft_rejected,  # type: ignore[arg-type]
    )
    assert "rejected" in str(ft_reject.get("error", "")).lower()

    # 4. ADK FunctionTool.run_async with ToolConfirmation(confirmed=True) publishes!
    ctx_ft_approved = _StubToolContext(
        function_call_id="call_ft_03",
        tool_confirmation=ToolConfirmation(
            confirmed=True,
            payload={"publish_id": staged_id},
        ),
    )
    ft_pub = await publish_to_linkedin_tool.run_async(
        args={"post_content": draft, "user_id": "u_adk"},
        tool_context=ctx_ft_approved,  # type: ignore[arg-type]
    )
    val_ft_pub = PublishFeedResponse.model_validate(ft_pub)
    assert val_ft_pub.status == "published"
    assert val_ft_pub.publish_id == staged_id


# ==============================================================================
# G7. Iteration 2 Adversarial Regression Suite (Fixes 1–7)
# ==============================================================================


@pytest.mark.asyncio
async def test_g7_fix1_concurrent_25_saves_and_slow_embed_never_overwrite_newer_profile(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "isolated_thrilled_data"
    store = get_default_style_store(data_dir=data_dir)

    # 25 rapid concurrent async saves for the same user + mixed idiom examples
    for idx in range(25):
        res = save_wording_preference(
            user_id="user_concurrent_25",
            raw_instruction=f"Never use 'word{idx}' in my posts—save this preference",
            approved_human_example=(
                f"Candid engineering note {idx}" if idx % 4 == 0 else None
            ),
            approved_linkedin_example=(
                f"Polished engineering update {idx}" if idx % 4 == 0 else None
            ),
        )
        validated = SavePreferenceResponse.model_validate(res)
        assert validated.status == "ok"
        assert validated.persistence_mode == "async_background_queued"

    # Interleave distinct-user saves on the same SQLite store
    for idx in range(5):
        save_wording_preference(
            user_id=f"peer_user_{idx}",
            raw_instruction=f"Never use 'peerword{idx}'—save this preference",
            preferred_phrases=[f"engineering excellence {idx}"],
        )

    await store.drain_background_tasks()
    await drain_all_style_tasks()

    cold_store = VectorStyleMemoryStore(data_dir=data_dir)
    cold_profile = cold_store.get_profile("user_concurrent_25")
    expected_words = [f"word{idx}" for idx in range(25)]
    assert cold_profile.banned_words == expected_words
    assert len(cold_store.search_similar_examples("user_concurrent_25", "engineering", top_k=10)) == 7
    for idx in range(5):
        assert cold_store.get_profile(f"peer_user_{idx}").banned_words == [
            f"peerword{idx}"
        ]

    # Out-of-order delayed embedding race: Task 1 sleeps 50ms while Task 2 (eval-C05) commits immediately
    class SlowEmbedIndex(DenseVectorSearchIndex):
        async def embed_text_async(self, text: str) -> list[float]:
            await asyncio.sleep(0.05)
            return self.embed_text_deterministic(text)

    race_dir = tmp_path / "race_store"
    race_store = VectorStyleMemoryStore(
        data_dir=race_dir,
        vector_index=SlowEmbedIndex(),
    )
    race_store.schedule_async_consolidation(
        user_id="race_user",
        raw_instruction="Make it shorter and avoid 'exhausted'—save this preference",
        approved_human_example="I quit my job",
        approved_linkedin_example="Stepping into a new engineering chapter.",
    )
    race_store.schedule_async_consolidation(
        user_id="race_user",
        raw_instruction="Actually, switch my default style to upbeat bullet points",
    )
    await race_store.drain_background_tasks()

    cold_race = VectorStyleMemoryStore(data_dir=race_dir)
    cold_race_prof = cold_race.get_profile("race_user")
    assert cold_race_prof.preferred_format == PreferredFormat.BULLETS
    assert cold_race_prof.tone_summary == "upbeat, bullet-point format"
    assert cold_race_prof.banned_words == ["exhausted"]
    assert len(cold_race.search_similar_examples("race_user", "chapter", top_k=3)) == 1


def test_g7_fix1_loop_safe_persist_lock_across_separate_asyncio_runs(
    tmp_path: Path,
) -> None:
    loop_dir = tmp_path / "multi_loop_store"
    shared_store = VectorStyleMemoryStore(data_dir=loop_dir)

    async def _first_loop_run() -> None:
        shared_store.schedule_async_consolidation(
            user_id="loop_user",
            raw_instruction="Never use 'synergy'—save this preference",
        )
        await shared_store.drain_background_tasks()

    async def _second_loop_run() -> None:
        shared_store.schedule_async_consolidation(
            user_id="loop_user",
            raw_instruction="Actually, switch my default style to upbeat bullet points",
        )
        await shared_store.drain_background_tasks()

    asyncio.run(_first_loop_run())
    asyncio.run(_second_loop_run())

    reloaded = VectorStyleMemoryStore(data_dir=loop_dir).get_profile("loop_user")
    assert reloaded.preferred_format == PreferredFormat.BULLETS
    assert reloaded.banned_words == ["synergy"]


@pytest.mark.asyncio
async def test_g7_fix2_multiturn_session_memory_deduplicates_events_and_respects_tool_calls(
    tmp_path: Path,
) -> None:
    mem_dir = tmp_path / "multiturn_mem"
    mem_service = PersistentVectorMemoryService(data_dir=mem_dir)

    @dataclass
    class _CbCtx:
        session: Session
        memory_service: PersistentVectorMemoryService

    # Standalone multi-turn replay: Turn 1 bans emojis + 'thrilled', Turn 2 re-enables both
    turn1_ev = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Never use 'thrilled', 'humbled', or emojis in my posts—save that rule"
                )
            ],
        ),
    )
    sess = Session(
        id="sess_mt_standalone",
        app_name="thrilled_to_announce",
        user_id="u_mt_1",
        state={},
        events=[turn1_ev],
    )
    after_agent_memory_callback(_CbCtx(session=sess, memory_service=mem_service))
    await drain_memory_tasks()
    assert mem_service.style_store.get_profile("u_mt_1").avoid_emojis is True
    assert mem_service.style_store.get_profile("u_mt_1").banned_words == [
        "thrilled",
        "humbled",
    ]

    turn2_ev = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[
                types.Part.from_text(
                    text="Actually allow emojis and allow 'thrilled' in my posts"
                )
            ],
        ),
    )
    sess.events.append(turn2_ev)
    after_agent_memory_callback(_CbCtx(session=sess, memory_service=mem_service))
    await drain_memory_tasks()

    cold_prof_1 = VectorStyleMemoryStore(data_dir=mem_dir).get_profile("u_mt_1")
    assert cold_prof_1.avoid_emojis is False
    assert cold_prof_1.banned_words == ["humbled"]

    # Tool-assisted multi-turn replay: Turn 1 sets concise=True, Turn 2 calls save_wording_preference(keep_concise=False)
    t1_short = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[types.Part.from_text(text="Make it shorter—save this preference")],
        ),
    )
    sess2 = Session(
        id="sess_mt_tool",
        app_name="thrilled_to_announce",
        user_id="u_mt_2",
        state={},
        events=[t1_short],
    )
    await mem_service.add_session_to_memory(sess2)
    assert mem_service.style_store.get_profile("u_mt_2").keep_concise is True

    # Turn 2 updates store via tool and logs function_call event in session
    mem_service.style_store.schedule_async_consolidation(
        user_id="u_mt_2",
        raw_instruction="Make my posts longer from now on",
        keep_concise=False,
    )
    await mem_service.style_store.drain_background_tasks()
    t2_user = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[types.Part.from_text(text="Make my posts longer from now on")],
        ),
    )
    t2_tool_call = Event(
        author="coordinator_agent",
        content=types.Content(
            role="model",
            parts=[
                types.Part.from_function_call(
                    name="save_wording_preference",
                    args={
                        "user_id": "u_mt_2",
                        "raw_instruction": "Make my posts longer from now on",
                        "keep_concise": False,
                    },
                )
            ],
        ),
    )
    sess2.events.extend([t2_user, t2_tool_call])
    await mem_service.add_session_to_memory(sess2)
    await mem_service.add_events_to_memory(
        app_name="thrilled_to_announce",
        user_id="u_mt_2",
        events=sess2.events,
        session_id=sess2.id,
    )
    cold_prof_2 = VectorStyleMemoryStore(data_dir=mem_dir).get_profile("u_mt_2")
    assert cold_prof_2.keep_concise is False


@pytest.mark.parametrize(
    ("instruction", "expected_banned"),
    [
        ("Avoid 'exhausted' and sound human", ["exhausted"]),
        ("Avoid 'exhausted', keep concise", ["exhausted"]),
        ("Avoid 'exhausted' and stay upbeat", ["exhausted"]),
        ("Never use 'thrilled' please", ["thrilled"]),
        ("Never use 'thrilled' on LinkedIn", ["thrilled"]),
        ("Never use 'thrilled' again", ["thrilled"]),
        (
            "Avoid sounding robotic, avoid corporate jargon, and never use \u2018thrilled\u2019 or \u201csynergy\u201d",
            ["thrilled", "synergy"],
        ),
        (
            "Remove 'Thrilled to announce' from the opening sentence",
            ["thrilled to announce"],
        ),
        ("Avoid 'exhausted' and use 'momentum'", ["exhausted"]),
    ],
)
def test_g7_fix3_compound_ban_clauses_and_trailing_modifiers_do_not_bleed(
    instruction: str, expected_banned: list[str]
) -> None:
    prof, _ = parse_and_merge_preferences(
        WordingPreferenceProfile(user_id="u_clause"),
        instruction,
    )
    assert prof.banned_words == expected_banned

    # Verify downstream validator accepts normal words like 'again' and 'please'
    clean_draft = "Excited to collaborate with backend engineers again this quarter."
    res_val = score_tone_and_bridge_burn_risk(
        candidate_text=clean_draft,
        target_style="LINKEDIN_SPEAK",
        banned_words=prof.banned_words,
    )
    assert res_val["status"] == "ok"


def test_g7_fix3_unbanning_and_punctuation_only_unit_norm_vectors() -> None:
    seed = WordingPreferenceProfile(
        user_id="u_unban",
        banned_words=["thrilled", "synergy", "humbled"],
        has_custom_preferences=True,
    )
    p1, _ = parse_and_merge_preferences(seed, "Actually you can use 'thrilled'")
    assert p1.banned_words == ["synergy", "humbled"]

    p2, _ = parse_and_merge_preferences(p1, "Remove 'synergy' from banned words")
    assert p2.banned_words == ["humbled"]

    idx = DenseVectorSearchIndex(dim=128)
    for punct_only in ("!!! ??? ...", "--- *** ///", "!"):
        vec = idx.embed_text_deterministic(punct_only)
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-5


@pytest.mark.asyncio
async def test_g7_fix4_strict_monotonic_hitl_blocks_forged_cancelled_and_non_bool() -> None:
    draft = "Proud to share lessons from scaling our distributed storage service."

    # 1. Forged/unknown publish_id in ToolConfirmation NEVER falls through to 'published'
    ctx_forged = _StubToolContext(
        function_call_id="fc_forged",
        tool_confirmation=ToolConfirmation(
            confirmed=True,
            payload={"publish_id": "li_post_forged_deadbeef"},
        ),
    )
    res_forged = await publish_to_linkedin_tool.run_async(
        args={"post_content": draft, "user_id": "u_hitl_strict"},
        tool_context=ctx_forged,  # type: ignore[arg-type]
    )
    val_forged = PublishFeedResponse.model_validate(res_forged)
    assert val_forged.status == "recoverable_error"
    assert get_default_style_store().get_pending_publish("li_post_forged_deadbeef") is None

    # 2. Empty/non-string publish_id in payload rejected
    ctx_empty_pid = _StubToolContext(
        function_call_id="fc_empty_pid",
        tool_confirmation=ToolConfirmation(confirmed=True, payload={"publish_id": "   "}),
    )
    res_empty_pid = publish_to_linkedin_feed(
        draft, "u_hitl_strict", tool_context=ctx_empty_pid  # type: ignore[arg-type]
    )
    assert PublishFeedResponse.model_validate(res_empty_pid).status == "recoverable_error"

    # 3. Cancelled publish_id is strictly terminal — replay via ToolConfirmation rejected
    staged = request_publish_approval(post_content=draft, user_id="u_hitl_strict")
    staged_pid = staged["publish_id"]
    assert confirm_publish_decision(staged_pid, confirmed=False)["status"] == "cancelled_by_human"

    ctx_replay = _StubToolContext(
        function_call_id="fc_replay_cancel",
        tool_confirmation=ToolConfirmation(
            confirmed=True,
            payload={"publish_id": staged_pid},
        ),
    )
    res_replay = await publish_to_linkedin_tool.run_async(
        args={"post_content": draft, "user_id": "u_hitl_strict"},
        tool_context=ctx_replay,  # type: ignore[arg-type]
    )
    assert PublishFeedResponse.model_validate(res_replay).status == "recoverable_error"
    stored_after_replay = get_default_style_store().get_pending_publish(staged_pid)
    assert stored_after_replay is not None
    assert stored_after_replay["status"] == "cancelled_by_human"

    # 4. Published publish_id is strictly terminal — retroactive cancel & duplicate confirm rejected
    staged2 = request_publish_approval(post_content=draft, user_id="u_hitl_strict")
    pub_pid = staged2["publish_id"]
    assert confirm_publish_decision(pub_pid, confirmed=True)["status"] == "published"

    retro_cancel = confirm_publish_decision(pub_pid, confirmed=False)
    assert PublishFeedResponse.model_validate(retro_cancel).status == "recoverable_error"
    dup_pub = confirm_publish_decision(pub_pid, confirmed=True)
    assert PublishFeedResponse.model_validate(dup_pub).status == "recoverable_error"
    stored_pub = get_default_style_store().get_pending_publish(pub_pid)
    assert stored_pub is not None and stored_pub["status"] == "published"

    # 5. Non-boolean confirmed="false" rejected (never coerced via bool("false") == True)
    staged3 = request_publish_approval(post_content=draft, user_id="u_hitl_strict")
    bad_bool_res = confirm_publish_decision(
        staged3["publish_id"],
        confirmed="false",  # type: ignore[arg-type]
    )
    assert PublishFeedResponse.model_validate(bad_bool_res).status == "recoverable_error"

    ctx_str_bool = SimpleNamespace(
        function_call_id="fc_str_bool",
        tool_confirmation=SimpleNamespace(confirmed="false", payload=None),
    )
    res_str_bool = publish_to_linkedin_feed(
        draft,
        "u_hitl_strict",
        tool_context=ctx_str_bool,  # type: ignore[arg-type]
    )
    assert PublishFeedResponse.model_validate(res_str_bool).status == "recoverable_error"


def test_g7_fix5_url_anchors_excluded_from_hashtags_and_buzzwords() -> None:
    # 5 real hashtags + 1 URL fragment => allowed (<= 5 hashtags)
    draft_with_url_and_five_tags = (
        "Read the architecture guide at https://example.com/docs/#overview "
        "#cloud #python #systems #backend #engineering"
    )
    res_ok = score_tone_and_bridge_burn_risk(
        candidate_text=draft_with_url_and_five_tags,
        target_style="LINKEDIN_SPEAK",
    )
    val_ok = ToneValidationResponse.model_validate(res_ok)
    assert val_ok.status == "ok"
    assert val_ok.flagged_terms == []

    # 6 URLs with #anchors and 0 real hashtags => allowed
    six_urls = (
        "References: https://example.com/d1/#intro https://example.com/d2/#design "
        "https://example.com/d3/#metrics https://example.com/q3-report/#rollout "
        "https://example.com/d5/#security www.example.com/d6/#appendix"
    )
    res_urls = score_tone_and_bridge_burn_risk(
        candidate_text=six_urls,
        target_style="LINKEDIN_SPEAK",
    )
    assert ToneValidationResponse.model_validate(res_urls).status == "ok"

    # URL containing #hiring does not misclassify candid human input as LINKEDIN_SPEAK
    res_det = detect_human_or_linkedin_style(
        message="I applied on https://acme.example.com/careers/#hiring and never heard back",
        has_previous_draft=False,
    )
    assert DetectStyleResponse.model_validate(res_det).detected_style == DetectedStyle.HUMAN_DAILY


def test_g7_fix6_negated_publish_bare_post_and_remove_buzzword_tweak_detection() -> None:
    for non_publish in (
        "Don't publish this to LinkedIn yet!",
        "Do not publish that post to my LinkedIn feed",
        "Wait—never publish anything to LinkedIn without my review",
        "Should I publish this to LinkedIn or wait until Monday?",
        "Can I post this to LinkedIn?",
    ):
        res = detect_human_or_linkedin_style(non_publish, has_previous_draft=True)
        val = DetectStyleResponse.model_validate(res)
        assert val.status == "ok"
        assert val.detected_style != DetectedStyle.PUBLISH_REQUEST

    for bare_publish in (
        "Post to LinkedIn",
        "Please post to my LinkedIn feed",
    ):
        res_pub = detect_human_or_linkedin_style(bare_publish, has_previous_draft=True)
        val_pub = DetectStyleResponse.model_validate(res_pub)
        assert val_pub.status == "ok"
        assert val_pub.detected_style == DetectedStyle.PUBLISH_REQUEST

    res_rm = detect_human_or_linkedin_style(
        "Remove 'Thrilled to announce' from the opening sentence",
        has_previous_draft=True,
    )
    val_rm = DetectStyleResponse.model_validate(res_rm)
    assert val_rm.status == "ok"
    assert val_rm.detected_style == DetectedStyle.STYLE_TWEAK


def test_g7_fix7_bare_string_or_bytes_rejected_for_list_parameters() -> None:
    res_tone_str = score_tone_and_bridge_burn_risk(
        candidate_text="I built a reliable backend service for our platform.",
        target_style="LINKEDIN_SPEAK",
        banned_words="ai",  # type: ignore[arg-type]
    )
    val_tone_str = ToneValidationResponse.model_validate(res_tone_str)
    assert val_tone_str.status == "recoverable_error"
    assert val_tone_str.flagged_terms == ["validation_parameter_error"]

    res_tone_bytes = score_tone_and_bridge_burn_risk(
        candidate_text="I built a reliable backend service for our platform.",
        target_style="LINKEDIN_SPEAK",
        banned_words=b"ai",  # type: ignore[arg-type]
    )
    val_tone_bytes = ToneValidationResponse.model_validate(res_tone_bytes)
    assert val_tone_bytes.status == "recoverable_error"
    assert val_tone_bytes.flagged_terms == ["validation_parameter_error"]

    res_save_bw = save_wording_preference(
        user_id="u_str_bw",
        raw_instruction="Save my preference rule",
        banned_words="ai",  # type: ignore[arg-type]
    )
    val_save_bw = SavePreferenceResponse.model_validate(res_save_bw)
    assert val_save_bw.status == "recoverable_error"
    assert val_save_bw.persistence_mode == "none"

    res_save_pp = save_wording_preference(
        user_id="u_str_pp",
        raw_instruction="Save my preference rule",
        preferred_phrases="hello",  # type: ignore[arg-type]
    )
    val_save_pp = SavePreferenceResponse.model_validate(res_save_pp)
    assert val_save_pp.status == "recoverable_error"
    assert val_save_pp.persistence_mode == "none"
