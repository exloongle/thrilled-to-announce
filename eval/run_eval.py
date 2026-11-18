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
"""Automated 26-case Golden Dataset evaluation runner for Thrilled to Announce."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any

import httpx

from thrilled_to_announce.config.settings import reset_settings_cache
from thrilled_to_announce.observability import (
    clear_recorded_spans,
    clear_telemetry_logs,
)

_HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*")

_JUDGE_PROMPT = (
    "You are grading a Human <-> LinkedIn translation app.\n"
    "User input:\n{input}\n\n"
    "App reply:\n{reply}\n\n"
    "Expected meaning: the reply should clearly convey at least one of these ideas "
    "(exact wording NOT required, paraphrases count): {concepts}.\n"
    'Answer with JSON only: {{"conveys": true|false, "why": "<short reason>"}}'
)


class SemanticJudge:
    """LLM-as-judge for live runs: checks meaning instead of exact substrings."""

    def __init__(self, *, model: str, project: str, location: str) -> None:
        from google import genai

        token = os.environ.get("EVAL_JUDGE_ACCESS_TOKEN", "").strip()
        credentials = None
        if token:
            from google.oauth2.credentials import Credentials

            credentials = Credentials(token=token)
        self._model = model
        self._client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=credentials,
        )

    async def conveys(self, *, user_input: str, reply: str, concepts: list[str]) -> tuple[bool, str]:
        from google.genai import types

        prompt = _JUDGE_PROMPT.format(
            input=user_input, reply=reply, concepts=", ".join(repr(c) for c in concepts)
        )
        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=types.Schema(
                        type=types.Type.OBJECT,
                        properties={
                            "conveys": types.Schema(type=types.Type.BOOLEAN),
                            "why": types.Schema(type=types.Type.STRING),
                        },
                        required=["conveys", "why"],
                    ),
                    thinking_config=types.ThinkingConfig(
                        thinking_level=types.ThinkingLevel.LOW
                    ),
                    max_output_tokens=2048,
                ),
            )
            raw = (resp.text or "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            parsed = json.loads(match.group(0) if match else "{}")
            return bool(parsed.get("conveys")), str(parsed.get("why", ""))[:200]
        except Exception as exc:  # judge failures never mask a real miss
            return False, f"judge_error: {type(exc).__name__}"


def _build_judge(mode: str) -> SemanticJudge | None:
    if mode != "live" or os.environ.get("EVAL_SEMANTIC_JUDGE", "TRUE").upper() == "FALSE":
        return None
    project = os.environ.get("EVAL_JUDGE_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        return None
    return SemanticJudge(
        model=os.environ.get("EVAL_JUDGE_MODEL", "gemini-4-hydrogen"),
        project=project,
        location=os.environ.get("EVAL_JUDGE_LOCATION", "global"),
    )


async def run_golden_suite_async(
    dataset: dict[str, Any],
    *,
    mode: str = "offline",
    base_url: str | None = None,
) -> dict[str, Any]:
    """Execute all 26 golden cases against the real FastAPI + ADK application."""
    cases = dataset.get("cases", [])
    thresholds = dataset.get("thresholds", {})
    if not cases:
        raise ValueError("Golden dataset contains no evaluation cases.")

    passed = 0
    failed = 0
    direction_hits = 0
    direction_total = 0
    per_case_results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="thrilled_eval_") as tmp_dir:
        prev_data_dir = os.environ.get("THRILLED_DATA_DIR")
        prev_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")
        if mode == "offline":
            os.environ["THRILLED_DATA_DIR"] = tmp_dir
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
            reset_settings_cache()
            clear_recorded_spans()
            clear_telemetry_logs()

        try:
            judge = _build_judge(mode)
            if base_url:
                async with httpx.AsyncClient(
                    base_url=base_url.rstrip("/"),
                    timeout=60.0,
                ) as client:
                    return await _execute_cases(
                        client, cases, thresholds, per_case_results, judge=judge
                    )

            from server.main import create_app

            app = create_app()
            async with app.router.lifespan_context(app):
                transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                    trust_env=False,
                ) as client:
                    return await _execute_cases(
                        client, cases, thresholds, per_case_results
                    )
        finally:
            if prev_data_dir is None:
                os.environ.pop("THRILLED_DATA_DIR", None)
            else:
                os.environ["THRILLED_DATA_DIR"] = prev_data_dir
            if prev_vertex is None:
                os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)
            else:
                os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = prev_vertex
            reset_settings_cache()

    return {
        "passed": passed,
        "failed": failed,
        "total": len(cases),
        "direction_hits": direction_hits,
        "direction_total": direction_total,
    }


async def _execute_cases(
    client: httpx.AsyncClient,
    cases: list[dict[str, Any]],
    thresholds: dict[str, Any],
    per_case_results: list[dict[str, Any]],
    *,
    judge: SemanticJudge | None = None,
) -> dict[str, Any]:
    """Run the 26-case suite sequentially over HTTP/ASGI."""
    passed = 0
    failed = 0
    direction_hits = 0
    direction_total = 0
    semantic_notes: dict[str, str] = {}

    seed_resp = await client.post(
        "/api/chat",
        json={
            "message": "I am tired of my current company and just wanted to find something not that boring",
            "user_id": "eval_user_c",
        },
    )
    seed_session = seed_resp.json()["session_id"]

    for case in cases:
        cid = case["id"]
        ok = True
        failure_reasons: list[str] = []

        if case.get("kind") == "welcome":
            res = await client.get("/api/welcome")
            body = res.json()
            if res.status_code != 200 or len(body.get("starter_prompts", [])) != 4:
                ok = False
                failure_reasons.append("welcome endpoint mismatch")
        elif case.get("kind") == "telemetry":
            res = await client.get("/api/telemetry")
            body = res.json()
            phases = {entry.get("phase") for entry in body.get("logs", [])}
            if res.status_code != 200 or not {"INTENT", "OUTCOME"}.issubset(phases):
                ok = False
                failure_reasons.append("missing paired INTENT/OUTCOME logs")
            if not body.get("spans"):
                ok = False
                failure_reasons.append("missing OpenTelemetry spans")
        else:
            uid = "eval_user_c" if cid.startswith("eval-C") or cid == "eval-F02" else f"u_{cid}"
            sid = seed_session if cid in {"eval-C02", "eval-F02"} else None

            res = await client.post(
                "/api/chat",
                json={
                    "message": case["input"],
                    "user_id": uid,
                    "session_id": sid,
                },
            )
            body = res.json()
            if res.status_code != 200:
                ok = False
                failure_reasons.append(f"HTTP {res.status_code}")

            if "expected_style" in case:
                direction_total += 1
                if body.get("detected_style") == case["expected_style"]:
                    direction_hits += 1
                else:
                    ok = False
                    failure_reasons.append(
                        f"expected_style={case['expected_style']} got={body.get('detected_style')}"
                    )

            reply_text = str(body.get("reply_text", ""))
            lower_reply = reply_text.lower()

            if case.get("expected_guardrail_block") and not body.get("guardrail_blocked"):
                ok = False
                failure_reasons.append("expected guardrail_blocked=True")

            if (
                "min_pii_redactions" in case
                and body.get("pii_redactions_count", 0) < case["min_pii_redactions"]
            ):
                ok = False
                failure_reasons.append("insufficient pii_redactions_count")

            if (
                "max_risk" in case
                and body.get("bridge_burn_risk_score", 100) > case["max_risk"]
            ):
                ok = False
                failure_reasons.append("bridge_burn_risk_score exceeded max_risk")

            if (
                "max_hashtags" in case
                and len(_HASHTAG_RE.findall(reply_text)) > case["max_hashtags"]
            ):
                ok = False
                failure_reasons.append("hashtag count exceeded max_hashtags")

            if case.get("expect_preference_saved") and not body.get("preference_saved"):
                ok = False
                failure_reasons.append("expected preference_saved=True")

            if case.get("expect_used_saved_preference") and not body.get(
                "used_saved_preference"
            ):
                ok = False
                failure_reasons.append("expected used_saved_preference=True")

            if case.get("expect_hitl_required") and not body.get("hitl_required"):
                ok = False
                failure_reasons.append("expected hitl_required=True")

            for bad in case.get("forbidden_substrings", []):
                if bad.lower() in lower_reply:
                    ok = False
                    failure_reasons.append(f"forbidden substring {bad!r} present")

            req_any = case.get("required_any_substrings", [])
            if req_any and not any(tok.lower() in lower_reply for tok in req_any):
                judged_ok, why = (False, "")
                if judge is not None and reply_text.strip():
                    judged_ok, why = await judge.conveys(
                        user_input=case["input"], reply=reply_text, concepts=req_any
                    )
                if judged_ok:
                    semantic_notes[cid] = f"semantic judge: {why}"
                else:
                    ok = False
                    failure_reasons.append(f"missing semantic stem from {req_any}")
                    if why:
                        failure_reasons.append(why)

        if ok:
            passed += 1
        else:
            failed += 1

        per_case_results.append(
            {
                "id": cid,
                "passed": ok,
                "reasons": failure_reasons,
                **({"note": semantic_notes[cid]} if cid in semantic_notes else {}),
            }
        )

    total = len(cases)
    direction_accuracy = direction_hits / max(direction_total, 1)
    overall_rate = passed / max(total, 1)
    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "layers": {
            "direction_accuracy": direction_accuracy,
            "deterministic_checks_pass_rate": overall_rate,
            "telemetry_safety_pass_rate": 1.0 if failed == 0 else overall_rate,
            "rubric_judge_ready": True,
        },
        "all_thresholds_met": (
            failed == 0
            and direction_accuracy >= float(thresholds.get("direction_accuracy", 1.0))
            and overall_rate >= float(thresholds.get("overall_pass_rate", 1.0))
        ),
        "cases": per_case_results,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for `python -m eval.run_eval`."""
    parser = argparse.ArgumentParser(
        prog="eval.run_eval",
        description="Run the 26-case Thrilled to Announce golden evaluation suite.",
    )
    parser.add_argument(
        "--dataset",
        default="eval/golden_dataset.json",
        help="Path to golden_dataset.json.",
    )
    parser.add_argument(
        "--mode",
        choices=("offline", "live"),
        default="offline",
        help="Execution mode (offline ASGI or live HTTP).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Optional live HTTP URL when running in live mode.",
    )
    args = parser.parse_args(argv)

    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        sys.stderr.write(f"Dataset file not found: {dataset_path}\n")
        return 1

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    summary = asyncio.run(
        run_golden_suite_async(
            dataset,
            mode=args.mode,
            base_url=args.base_url,
        )
    )
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0 if summary.get("all_thresholds_met") else 1


if __name__ == "__main__":
    sys.exit(main())
