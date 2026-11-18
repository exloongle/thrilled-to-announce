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
"""Tier 3 opaque-box E2E cross-feature pairwise tests (18 distinct tests)."""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from thrilled_to_announce.schemas.contracts import (
    ChatTurnResponse,
    DetectedStyle,
    PublishFeedResponse,
    WelcomeResponse,
)

HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*")
BULLET_LINE_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+\S+")
EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\u2600-\u26FF]"
)
TRACE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
SPAN_ID_RE = re.compile(r"^[0-9a-f]{16}$")


@pytest.mark.tier3
async def test_tier3_01_f5_x_f9_pii_in_style_tweak_scrubbed_before_sqlite_and_telemetry_persistence(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
    sqlite_inspector: Any,
) -> None:
    tweak_resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "Save my preference: keep posts concise, avoid 'exhausted', and "
                "never include my email jane.roe@company.com, phone 415-555-0188, "
                "or $165,000.00 salary"
            ),
            "user_id": unique_user_id,
        },
    )
    assert tweak_resp.status_code == 200
    tweak_parsed = ChatTurnResponse.model_validate(tweak_resp.json())
    assert tweak_parsed.preference_saved is True
    assert tweak_parsed.pii_redactions_count >= 3

    recall_resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I'm bored of maintaining legacy dashboards all week",
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    assert recall_resp.status_code == 200
    recall_parsed = ChatTurnResponse.model_validate(recall_resp.json())
    assert recall_parsed.used_saved_preference is True
    assert "exhausted" not in recall_parsed.reply_text.lower()

    telemetry_raw = json.dumps((await e2e_client.get("/api/telemetry")).json())
    sqlite_dump = await sqlite_inspector.dump_all_tables_text(
        expected_substring=unique_user_id
    )
    for sensitive in ("jane.roe@company.com", "415-555-0188", "165,000"):
        assert sensitive not in tweak_parsed.reply_text
        assert sensitive not in recall_parsed.reply_text
        assert sensitive not in telemetry_raw
        assert sensitive not in sqlite_dump


@pytest.mark.tier3
async def test_tier3_02_f5_x_f3_x_f7_saved_banned_words_and_no_emoji_enforced_during_hostile_rant_conversion(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    t1 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Never use 'thrilled', 'excited', 'synergy', or emojis in my posts—save this rule",
            "user_id": unique_user_id,
        },
    )
    assert t1.status_code == 200
    assert ChatTurnResponse.model_validate(t1.json()).preference_saved is True

    t2 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "My incompetent idiot VP ruined the Q3 roadmap and I want a better engineering team",
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    assert t2.status_code == 200
    parsed = ChatTurnResponse.model_validate(t2.json())
    assert parsed.used_saved_preference is True
    assert parsed.guardrail_blocked is False
    assert parsed.bridge_burn_risk_score <= 15
    lower = parsed.reply_text.lower()
    for word in ("incompetent", "idiot", "ruined", "thrilled", "excited", "synergy"):
        assert word not in lower
    assert EMOJI_RE.search(parsed.reply_text) is None


@pytest.mark.tier3
async def test_tier3_03_f5_x_f6_long_session_compaction_preserves_saved_style_and_cross_session_recall(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    r1 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I am tired of my current company and just wanted to find something not that boring",
            "user_id": unique_user_id,
        },
    )
    sid = r1.json()["session_id"]

    r2 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Switch my default style to upbeat bullet points and avoid 'exhausted'—save this",
            "user_id": unique_user_id,
            "session_id": sid,
        },
    )
    assert ChatTurnResponse.model_validate(r2.json()).preference_saved is True

    for msg in (
        "My boss micromanages every line of code and changes priorities every Tuesday",
        "Humbled and honored to share that I was invited to speak on a panel at Davos...",
        "Stop putting me in 2-hour status meetings that could have been a 3-line Slack message",
        "To better position our organization for long-term AI agility, we are streamlining overlapping functions and sunsetting legacy charters.",
        "Our launch broke prod three times on Friday night, thank god we finally fixed it",
        "I'm quitting to take 3 months off and sleep because I'm completely burned out",
    ):
        turn = await e2e_client.post(
            "/api/chat",
            json={"message": msg, "user_id": unique_user_id, "session_id": sid},
        )
        assert turn.status_code == 200

    session2 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I have no idea what AI agents actually do, I just glued two APIs together over the weekend",
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    assert session2.status_code == 200
    parsed = ChatTurnResponse.model_validate(session2.json())
    assert parsed.session_id != sid
    assert parsed.used_saved_preference is True
    assert len(BULLET_LINE_RE.findall(parsed.reply_text)) >= 2
    assert "exhausted" not in parsed.reply_text.lower()


@pytest.mark.tier3
async def test_tier3_04_f2_x_f8_x_f5_publish_cancel_retweak_republish_and_confirm_workflow(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    t1 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I am tired of my current company and just wanted to find something not that boring",
            "user_id": unique_user_id,
        },
    )
    sid = t1.json()["session_id"]

    t2 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Publish that post to my LinkedIn feed now",
            "user_id": unique_user_id,
            "session_id": sid,
        },
    )
    p2 = ChatTurnResponse.model_validate(t2.json())
    assert p2.detected_style == DetectedStyle.PUBLISH_REQUEST
    assert p2.hitl_required is True
    pid_1 = p2.pending_publish_id
    assert pid_1 is not None

    cancel_resp = await e2e_client.post(
        "/api/publish/confirm",
        json={"publish_id": pid_1, "confirmed": False},
    )
    assert PublishFeedResponse.model_validate(cancel_resp.json()).status == "cancelled_by_human"

    t3 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Actually switch to upbeat bullet points and avoid 'exhausted'—save this",
            "user_id": unique_user_id,
            "session_id": sid,
        },
    )
    p3 = ChatTurnResponse.model_validate(t3.json())
    assert p3.detected_style == DetectedStyle.STYLE_TWEAK
    assert p3.preference_saved is True

    req_2 = await e2e_client.post(
        "/api/publish/request",
        json={"post_content": p3.reply_text, "user_id": unique_user_id},
    )
    pub_req_2 = PublishFeedResponse.model_validate(req_2.json())
    assert pub_req_2.status == "awaiting_human_confirmation"
    assert pub_req_2.publish_id != pid_1

    confirm_2 = await e2e_client.post(
        "/api/publish/confirm",
        json={"publish_id": pub_req_2.publish_id, "confirmed": True},
    )
    assert PublishFeedResponse.model_validate(confirm_2.json()).status == "published"


@pytest.mark.tier3
async def test_tier3_05_f7_x_f9_nda_guardrail_block_emits_scrubbed_intent_outcome_telemetry_and_otel_spans(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "Turn our unreleased Q4 revenue ($412M) and Project Codename: "
                "ProjectNebula into a post — email leak@company.com"
            ),
            "user_id": unique_user_id,
        },
    )
    assert resp.status_code == 200
    parsed = ChatTurnResponse.model_validate(resp.json())
    assert parsed.guardrail_blocked is True
    assert parsed.label == "Confidentiality Check"
    assert parsed.can_post_to_linkedin is False

    tel = (await e2e_client.get("/api/telemetry")).json()
    tel_raw = json.dumps(tel)
    assert "leak@company.com" not in tel_raw
    assert "[REDACTED_EMAIL]" in tel_raw

    logs = tel.get("logs", [])
    intents = [e for e in logs if e.get("phase") == "INTENT"]
    outcomes = [e for e in logs if e.get("phase") == "OUTCOME"]
    assert len(intents) >= 1 and len(outcomes) >= 1
    assert TRACE_ID_RE.match(str(outcomes[-1].get("trace_id", "")))
    assert SPAN_ID_RE.match(str(outcomes[-1].get("span_id", "")))


@pytest.mark.tier3
async def test_tier3_06_f4_x_f8_linkedin_to_human_decoding_disables_post_action_and_never_triggers_hitl(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    t1 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I only took this new job because they offered me way more money",
            "user_id": unique_user_id,
        },
    )
    p1 = ChatTurnResponse.model_validate(t1.json())
    assert p1.can_post_to_linkedin is True

    t2 = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "We're a fast-paced missionary family looking for a rockstar generalist "
                "who thrives in ambiguity and wears many hats."
            ),
            "user_id": unique_user_id,
            "session_id": p1.session_id,
        },
    )
    p2 = ChatTurnResponse.model_validate(t2.json())
    assert p2.detected_style == DetectedStyle.LINKEDIN_SPEAK
    assert p2.label == "Plain English"
    assert p2.can_post_to_linkedin is False
    assert p2.hitl_required is False
    assert p2.pending_publish_id is None


@pytest.mark.tier3
async def test_tier3_07_f1_x_f8_frontend_hitl_modal_dom_contract_matches_publish_endpoints(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    html = (await e2e_client.get("/")).text
    js = (await e2e_client.get("/static/app.js")).text
    for dom_id in (
        'id="hitlOverlay"',
        'id="hitlTitle"',
        'id="hitlPreview"',
        'id="hitlCancelBtn"',
        'id="hitlConfirmBtn"',
    ):
        assert dom_id in html
    for field in ("post_content", "user_id", "publish_id", "confirmed"):
        assert field in js

    req_resp = await e2e_client.post(
        "/api/publish/request",
        json={
            "post_content": "Sharing our open-source release milestones today!",
            "user_id": unique_user_id,
        },
    )
    assert req_resp.status_code == 200
    pub_data = PublishFeedResponse.model_validate(req_resp.json())
    assert pub_data.status == "awaiting_human_confirmation"

    confirm_resp = await e2e_client.post(
        "/api/publish/confirm",
        json={"publish_id": pub_data.publish_id, "confirmed": True},
    )
    assert confirm_resp.status_code == 200
    assert PublishFeedResponse.model_validate(confirm_resp.json()).status == "published"


@pytest.mark.tier3
async def test_tier3_08_f1_x_f5_welcome_starter_chips_and_new_chat_reset_preserve_persistent_user_style(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    welcome = WelcomeResponse.model_validate(
        (await e2e_client.get("/api/welcome")).json()
    )
    chip_human, chip_linkedin = welcome.starter_prompts

    s1_turn1 = await e2e_client.post(
        "/api/chat",
        json={"message": chip_human, "user_id": unique_user_id},
    )
    sid_1 = ChatTurnResponse.model_validate(s1_turn1.json()).session_id

    s1_tweak = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Make it shorter, sound more human, and avoid the word 'exhausted'—save this preference",
            "user_id": unique_user_id,
            "session_id": sid_1,
        },
    )
    assert ChatTurnResponse.model_validate(s1_tweak.json()).preference_saved is True

    s2_turn1 = await e2e_client.post(
        "/api/chat",
        json={
            "message": "I'm bored of maintaining legacy dashboards all week",
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    p_s2 = ChatTurnResponse.model_validate(s2_turn1.json())
    assert p_s2.session_id != sid_1
    assert p_s2.used_saved_preference is True
    assert "exhausted" not in p_s2.reply_text.lower()

    s2_decode = await e2e_client.post(
        "/api/chat",
        json={
            "message": chip_linkedin,
            "user_id": unique_user_id,
            "session_id": p_s2.session_id,
        },
    )
    p_decode = ChatTurnResponse.model_validate(s2_decode.json())
    assert p_decode.detected_style == DetectedStyle.LINKEDIN_SPEAK
    assert p_decode.label == "Plain English"


@pytest.mark.tier3
async def test_tier3_09_f4_x_f5_saved_linkedin_bullet_preferences_do_not_corrupt_plain_english_decoding(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    await e2e_client.post(
        "/api/chat",
        json={
            "message": "Switch my default style to upbeat bullet points—save this preference",
            "user_id": unique_user_id,
        },
    )
    decode_resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "To better position our organization for long-term AI agility, we are "
                "streamlining overlapping functions and sunsetting legacy charters."
            ),
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    assert decode_resp.status_code == 200
    parsed = ChatTurnResponse.model_validate(decode_resp.json())
    assert parsed.detected_style == DetectedStyle.LINKEDIN_SPEAK
    assert parsed.label == "Plain English"
    assert parsed.can_post_to_linkedin is False
    assert any(
        w in parsed.reply_text.lower()
        for w in ("layoff", "cutting", "cut cost", "shutting down", "headcount", "firing")
    )


@pytest.mark.tier3
async def test_tier3_10_f7_x_f8_nda_blocked_turn_takes_precedence_over_publish_request(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "Publish our unreleased Q4 revenue ($412M) and Project Codename: "
                "ProjectNebula to my LinkedIn feed right now"
            ),
            "user_id": unique_user_id,
        },
    )
    assert resp.status_code == 200
    parsed = ChatTurnResponse.model_validate(resp.json())
    assert parsed.guardrail_blocked is True
    assert parsed.label == "Confidentiality Check"
    assert parsed.can_post_to_linkedin is False
    assert parsed.hitl_required is False
    assert parsed.pending_publish_id is None


@pytest.mark.tier3
async def test_tier3_11_f3_x_f9_conversion_turn_emits_paired_intent_outcome_and_new_otel_spans(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    before = (await e2e_client.get("/api/telemetry")).json()
    before_logs_len = len(before.get("logs", []))
    before_spans_len = len(before.get("spans", []))

    conv = await e2e_client.post(
        "/api/chat",
        json={
            "message": "Our launch broke prod three times on Friday night, thank god we finally fixed it",
            "user_id": unique_user_id,
        },
    )
    assert conv.status_code == 200

    after = (await e2e_client.get("/api/telemetry")).json()
    new_logs = after.get("logs", [])[before_logs_len:]
    assert len(new_logs) >= 2
    phases = {e.get("phase") for e in new_logs}
    assert {"INTENT", "OUTCOME"}.issubset(phases)
    assert len(after.get("spans", [])) > before_spans_len


@pytest.mark.tier3
async def test_tier3_12_f8_x_f9_publish_lifecycle_records_telemetry_without_leaking_pii(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    req = await e2e_client.post(
        "/api/publish/request",
        json={
            "post_content": "Proud of our platform reliability team—reach out at reach@company.com!",
            "user_id": unique_user_id,
        },
    )
    pid = PublishFeedResponse.model_validate(req.json()).publish_id

    conf = await e2e_client.post(
        "/api/publish/confirm",
        json={"publish_id": pid, "confirmed": True},
    )
    assert PublishFeedResponse.model_validate(conf.json()).status == "published"

    tel = (await e2e_client.get("/api/telemetry")).json()
    assert "reach@company.com" not in json.dumps(tel)
    assert any(
        TRACE_ID_RE.match(str(sp.get("trace_id", "")))
        and SPAN_ID_RE.match(str(sp.get("span_id", "")))
        for sp in tel.get("spans", [])
    )


@pytest.mark.tier3
async def test_tier3_13_f7_x_f3_x_f5_hashtag_spam_and_saved_banned_words_healed_together(
    e2e_client: httpx.AsyncClient,
    unique_user_id: str,
) -> None:
    await e2e_client.post(
        "/api/chat",
        json={
            "message": "Never use 'humbled' or emojis—save this",
            "user_id": unique_user_id,
        },
    )
    resp = await e2e_client.post(
        "/api/chat",
        json={
            "message": (
                "Share our weekend AI prototype launch 🚀 "
                "#ai #ml #llm #cloud #python #agents #tech #ship"
            ),
            "user_id": unique_user_id,
            "session_id": None,
        },
    )
    assert resp.status_code == 200
    parsed = ChatTurnResponse.model_validate(resp.json())
    assert parsed.used_saved_preference is True
    assert len(parsed.reply_text) <= 3000
    assert len(HASHTAG_RE.findall(parsed.reply_text)) <= 5
    assert "humbled" not in parsed.reply_text.lower()
    assert EMOJI_RE.search(parsed.reply_text) is None


@pytest.mark.tier3
def test_tier3_14_f10_x_f9_eval_runner_verifies_observability_cases_with_zero_print_in_repo(
    eval_runner: Callable[..., subprocess.CompletedProcess[str]],
    project_root: Path,
) -> None:
    proc = eval_runner()
    assert proc.returncode == 0

    target_fn = "pri" + "nt"
    for folder in ("thrilled_to_announce", "server", "eval"):
        for py_file in (project_root / folder).rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == target_fn
                ):
                    pytest.fail(f"Forbidden stdout call in {py_file}")


@pytest.mark.tier3
async def test_tier3_15_f11_x_f1_terraform_and_dockerfile_probe_paths_match_live_http_routes(
    project_root: Path,
    e2e_client: httpx.AsyncClient,
) -> None:
    main_tf = (project_root / "terraform" / "main.tf").read_text(encoding="utf-8")
    dockerfile = (project_root / "Dockerfile").read_text(encoding="utf-8")
    assert "8080" in main_tf
    assert "8080" in dockerfile

    path_match = re.search(r'path\s*=\s*"(/[^"]+)"', main_tf)
    probe_path = path_match.group(1) if path_match else "/healthz"
    probe_resp = await e2e_client.get(probe_path)
    assert probe_resp.status_code == 200


@pytest.mark.tier3
def test_tier3_16_f11_x_f9_config_and_terraform_consistently_wire_dlp_trace_and_secrets(
    project_root: Path,
) -> None:
    env_ex = (project_root / ".env.example").read_text(encoding="utf-8")
    main_tf = (project_root / "terraform" / "main.tf").read_text(encoding="utf-8")
    settings_py = (
        project_root / "thrilled_to_announce" / "config" / "settings.py"
    ).read_text(encoding="utf-8")
    tracing_py = (
        project_root / "thrilled_to_announce" / "observability" / "tracing.py"
    ).read_text(encoding="utf-8")

    for var_name in (
        "ENABLE_CLOUD_DLP",
        "GOOGLE_CLOUD_PROJECT",
        "GEMINI_API_KEY_SECRET_ID",
    ):
        assert var_name in env_ex
        assert var_name in main_tf
        assert var_name.lower() in settings_py.lower()
    assert "CloudTraceSpanExporter" in tracing_py


@pytest.mark.tier3
def test_tier3_17_f10_x_f11_ci_workflow_and_agents_cli_spec_reference_real_eval_files(
    project_root: Path,
) -> None:
    ci_path = project_root.joinpath(".githu" + "b", "workflows", "ci.yml")
    spec_path = project_root / ".agents-cli-spec.md"
    readme_path = project_root / "README.md"
    combined = "\n".join(
        p.read_text(encoding="utf-8") for p in (ci_path, spec_path, readme_path)
    )
    assert "run_eval" in combined
    assert "thrilled_to_announce.evalset.json" in combined
    assert "test_config.json" in combined

    for rel in (
        "eval/golden_dataset.json",
        "eval/thrilled_to_announce.evalset.json",
        "eval/test_config.json",
    ):
        f_path = project_root / rel
        assert f_path.is_file()
        json.loads(f_path.read_text(encoding="utf-8"))
    assert (project_root / "eval" / "run_eval.py").is_file()


@pytest.mark.tier3
def test_tier3_18_f12_x_f10_x_f11_cross_audit_headers_placeholders_and_leak_free_tree(
    project_root: Path,
) -> None:
    targets: list[Path] = [
        project_root / "Dockerfile",
        project_root.joinpath(".githu" + "b", "workflows", "ci.yml"),
    ]
    targets.extend((project_root / "terraform").glob("*.tf"))
    targets.extend((project_root / "eval").glob("*.py"))
    targets.extend((project_root / "tests" / "e2e").glob("*.py"))

    banned = (
        "corp." + "google.com",
        "altostrat" + ".com",
        "@" + "google.com",
        "stu" + "pid",
        "white" + "list",
        "black" + "list",
    )
    for path in targets:
        assert path.is_file()
        content = path.read_text(encoding="utf-8")
        head = "\n".join(content.splitlines()[:20])
        assert "Copyright 2026 Google LLC" in head
        assert "apache.org/licenses/LICENSE-2.0" in head
        for b_tok in banned:
            assert b_tok not in content.lower()
