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
"""FastAPI web server wiring ADK multi-agent runtime, memory, guardrails, and UI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import os
from pathlib import Path
import re
from typing import Any
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from google.adk.runners import Runner
from google.genai import types

from thrilled_to_announce.app import create_adk_app
from thrilled_to_announce.config.secrets import get_secret_resolver
from thrilled_to_announce.config.settings import get_settings
from thrilled_to_announce.memory import (
    create_memory_service,
    create_session_service,
    drain_memory_tasks,
    get_default_style_store,
)
from thrilled_to_announce.observability import (
    configure_logging,
    configure_tracing,
    get_default_scrubber,
    get_logger,
    get_recorded_spans,
    get_telemetry_logs,
)
from thrilled_to_announce.plugins import (
    IntentOutcomeContext,
    evaluate_input_policy,
)
from thrilled_to_announce.schemas.contracts import (
    ChatTurnRequest,
    ChatTurnResponse,
    DetectedStyle,
    PreferredFormat,
    PublishConfirmHttpRequest,
    PublishFeedRequest,
    PublishFeedResponse,
    WelcomeResponse,
    WordingPreferenceProfile,
)
from thrilled_to_announce.tools import (
    confirm_publish_decision,
    detect_human_or_linkedin_style,
    request_publish_approval,
    retrieve_wording_preferences,
    save_wording_preference,
    score_tone_and_bridge_burn_risk,
)

_HASHTAG_RE = re.compile(r"(?<!\w)#[A-Za-z][A-Za-z0-9_]*")
_EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B50\U00002B55"
    r"\U0000231A-\U0000231B\U000023E9-\U000023F3\U000023F8-\U000023FA"
    r"\U0001F1E6-\U0001F1FF]"
)

_WELCOME_PAYLOAD = WelcomeResponse(
    greeting=(
        "Hi! Send me a blunt thought and I'll turn it into a polished LinkedIn "
        "announcement—or paste a LinkedIn post and I'll tell you what it really means."
    ),
    starter_prompts=[
        "My boss micromanages every line of code and changes priorities every Tuesday—I need out",
        "I'm quitting to take 3 months off and sleep because 2-hour status meetings burned me out",
        (
            "Humbled and honored to announce that after 4 impactful years driving synergistic "
            "transformation, I'm stepping back to explore fractional advisory horizons!"
        ),
        (
            "Yesterday at 4:45 AM my 3-year-old asked why SaaS net dollar retention matters "
            "more than ARR. Everyone in the coffee shop clapped. Agree?"
        ),
    ],
)


def _decode_linkedin_subtext(scrubbed_text: str) -> str:
    """Decode sugar-coated LinkedIn buzzwords into concise Plain English."""
    lower = scrubbed_text.lower()
    if "fractional advisory" in lower or "stepping back to explore" in lower:
        return (
            "I got laid off or left without another offer, and I am actively "
            "looking for a full-time job—please DM me if you are hiring."
        )
    if (
        "rockstar generalist" in lower
        or "wears many hats" in lower
        or "thrives in ambiguity" in lower
        or "missionary family" in lower
    ):
        return (
            "We are understaffed with no clear roadmap and weak boundaries, so "
            "you will do three jobs for one salary and risk burnout."
        )
    if (
        "davos" in lower
        or "humbled and honored" in lower
        or "grateful, humbled" in lower
    ):
        return (
            "I am not remotely humble right now—I am proud of this status flex "
            "and showing off so everyone in my network notices."
        )
    if (
        "profile really stood out" in lower
        or "contract-to-hire onsite role" in lower
    ):
        return (
            "I did not read your profile; I sent a mass keyword-filter template "
            "blast to hundreds of engineers for a hard-to-fill role."
        )
    if (
        "ai agility" in lower
        or "streamlining overlapping" in lower
        or "sunsetting" in lower
        or "right-sizing" in lower
        or "holistic portfolio review" in lower
    ):
        return (
            "We are doing layoffs, cutting headcount, and shutting down projects "
            "to reduce costs."
        )
    if (
        "4:45 am" in lower
        or "net dollar retention" in lower
        or "coffee shop clapped" in lower
        or "candidate in 4 seconds" in lower
    ):
        return (
            "This made-up story never happened; I invented a dramatic fictional "
            "anecdote to farm engagement, likes, and comments."
        )
    if "culture is everything" in lower or "agree?" in lower:
        return (
            "This dramatic post is engagement bait designed to boost reach and "
            "collect comments in the LinkedIn feed."
        )
    return (
        "Stripping away the corporate buzzwords: the author wants professional "
        "credit and network visibility without stating the blunt facts."
    )


def _compose_linkedin_sentences(scrubbed_text: str) -> list[str]:
    """Generate career-safe LinkedIn phrasing preserving factual human intent."""
    lower = scrubbed_text.lower()
    if "tired of my current company" in lower or "not that boring" in lower:
        return [
            "I've exhausted the growth opportunities in my current role and am actively seeking a more stimulating and dynamic environment for my professional development.",
            "Looking forward to connecting with teams tackling ambitious technical and product challenges.",
        ]
    if "micromanag" in lower or "priorities every tuesday" in lower:
        return [
            "I do my best engineering work in high-ownership, outcome-driven teams built on clear strategic alignment.",
            "Empowering autonomous execution across a focused roadmap helps everyone ship stronger results.",
        ]
    if "laid off" in lower or "ran out of budget" in lower:
        return [
            "Following a recent organizational restructuring, my role was impacted today, and I am grateful for everything our team built together.",
            "I am energized to explore new senior roles where I can contribute experience and deliver immediate impact.",
        ]
    if "2-hour status meetings" in lower or "3-line slack" in lower:
        return [
            "Protecting deep-work focus through crisp written updates and async-first communication cadences accelerates engineering velocity.",
            "Clear asynchronous collaboration gives builders more uninterrupted time to ship reliable systems.",
        ]
    if "way more money" in lower or "only took this new job" in lower:
        return [
            "Delighted to share that I am joining a new team whose engineering scope, mission, and market impact align with my next chapter.",
            "Grateful for the journey so far and ready to build ahead.",
        ]
    if "broke prod" in lower or "friday night" in lower:
        return [
            "Proud of our engineering team's resilience during a complex Friday production launch.",
            "Through blameless incident recovery and strong observability, we restored customer reliability and captured lasting lessons.",
        ]
    if "glued two apis" in lower or "ai agents" in lower or "weekend" in lower:
        return [
            "Spent the weekend focused on rapid prototyping and pragmatic API integration to ship a working 0-to-1 AI agent workflow.",
            "Small hands-on experiments remain the fastest path from concept to production learning.",
        ]
    if "quitting to take 3 months" in lower or "burned out" in lower:
        return [
            "Stepping back for an intentional three-month career sabbatical to recharge, reflect, and explore emerging technology horizons.",
            "Grateful for the colleagues I have partnered with and looking forward to my next chapter.",
        ]
    if "legacy dashboards" in lower or "bored of maintaining" in lower:
        return [
            "Ready to channel my engineering experience beyond routine dashboard maintenance into building modern, high-impact platforms.",
            "Open to connecting with teams tackling ambitious architectural problems.",
        ]
    if "incompetent" in lower or "idiot" in lower or "ruined the roadmap" in lower:
        return [
            "Looking ahead to my next engineering leadership opportunity with a team united around clear strategic vision and collaborative roadmap alignment.",
            "Disciplined execution and shared direction make all the difference when scaling products.",
        ]
    if "[redacted_" in lower or "pay is too low" in lower:
        return [
            "Open to connecting regarding senior engineering roles aligned with technical scope, team growth, and market compensation.",
            "Feel free to reach out via direct message on LinkedIn to start a conversation.",
        ]
    if "promoted" in lower:
        return [
            "Grateful to step into my newly promoted engineering role and expand my technical scope alongside an exceptional team.",
            "Looking forward to mentoring teammates and delivering durable impact in this next chapter.",
        ]
    if "cut our ci build time" in lower or "35%" in lower:
        return [
            "Proud to share that our engineering team cut CI build times by 35% across 14 repositories this month.",
            "Investing in developer productivity compounds velocity across every release.",
        ]
    if "tokyo" in lower or "zürich" in lower:
        return [
            "Delighted to ship our Tokyo 東京 and Zürich API gateway migration—delivering noticeably lower latency across global regions.",
            "Grateful to the cross-regional platform engineering team that made this rollout smooth.",
        ]
    if (
        "hate myself" in lower
        or "help me" in lower
        or "imposter" in lower
        or "overwhelmed" in lower
        or "failing at" in lower
    ):
        return [
            "Deeply humbled by an intense season of radical self-reflection and owning my toughest growth edges out loud.",
            "Grateful to be part of a thoughtful community where raising a hand for candid mentorship and collaborative problem-solving helps everyone level up together.",
        ]
    if "hate my" in lower or "hate this" in lower or "sucks" in lower:
        return [
            "Candid moments of friction often sharpen clarity on where we create the highest long-term value.",
            "Focused on channeling lessons from challenging chapters into intentional, high-impact work ahead.",
        ]
    if "flaky deployment" in lower or "flaky ci" in lower:
        return [
            "Investing engineering focus into stabilizing CI and deployment automation pays off every week in calmer, more reliable releases.",
            "Strong release pipelines let teams spend energy on customer value rather than manual toil.",
        ]
    tokens = re.findall(r"[a-zA-Z]{4,}", lower)
    stop = {
        "that",
        "this",
        "with",
        "from",
        "have",
        "been",
        "just",
        "want",
        "need",
        "really",
        "very",
        "about",
        "some",
        "someone",
        "myself",
        "today",
        "every",
        "week",
        "company",
        "work",
        "hate",
    }
    themes = list(dict.fromkeys(tok for tok in tokens if tok not in stop))[:3]
    if themes:
        theme_phrase = " and ".join([", ".join(themes[:-1]), themes[-1]]) if len(themes) > 1 else themes[0]
        return [
            f"Taking a candid moment to reflect on {theme_phrase}—and how leaning into honest feedback and collaborative problem-solving turns tough stretches into durable growth.",
            "Always open to comparing notes with peers navigating similar inflection points.",
        ]
    return [
        "Taking a candid moment for honest reflection—and turning everyday challenges into fuel for intentional growth and stronger collaboration.",
        "Always grateful to learn alongside thoughtful peers.",
    ]


_TOPIC_HASHTAG_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("mentor", "help", "support", "feedback", "growth", "reflection", "learn"), "#GrowthMindset"),
    (("mentorship", "peers", "community", "network", "team"), "#Leadership"),
    (("engineering", "code", "architecture", "sprint", "pipeline", "ci", "api", "prod"), "#SoftwareEngineering"),
    (("career", "job", "role", "opportunity", "sabbatical", "chapter", "hiring"), "#CareerGrowth"),
    (("burnout", "recharge", "sleep", "balance", "async", "focus", "meetings"), "#WorkCulture"),
    (("ai", "agent", "llm", "automation"), "#AI"),
)


def _suggest_topical_hashtags(
    draft_text: str,
    source_text: str,
    *,
    max_count: int = 3,
) -> list[str]:
    """Generate relevant career/industry hashtags from draft and source themes."""
    combined = f"{source_text} {draft_text}".lower()
    chosen: list[str] = []
    for keywords, tag in _TOPIC_HASHTAG_MAP:
        if any(re.search(rf"\b{re.escape(kw)}\b", combined) for kw in keywords) and tag not in chosen:
            chosen.append(tag)
        if len(chosen) >= max_count:
            break
    for fallback_tag in ("#CareerGrowth", "#ContinuousLearning", "#TechCommunity"):
        if len(chosen) >= max_count:
            break
        if fallback_tag not in chosen:
            chosen.append(fallback_tag)
    return chosen[:max_count]


def _apply_followup_tweak_to_sentences(
    *,
    sentences: list[str],
    prior_draft: str,
    source_text: str,
    tweak_instruction: str,
) -> tuple[list[str], list[str]]:
    """Apply direct follow-up draft modifications (e.g., hashtags) offline."""
    lower = tweak_instruction.lower()
    extra_tags: list[str] = []
    explicit_tags = _HASHTAG_RE.findall(tweak_instruction)
    if explicit_tags:
        extra_tags.extend(explicit_tags)
    elif re.search(r"\b(?:add|include|append|insert|put|with|more)\b.*\b(?:hashtags?|tags?)\b|\b(?:hashtags?|tags?)\b", lower) and not re.search(r"\b(?:remove|no|without|drop|strip)\b.*\b(?:hashtags?|tags?)\b", lower):
        extra_tags.extend(_suggest_topical_hashtags(prior_draft, source_text, max_count=3))

    base_sentences = list(sentences)
    if not base_sentences and prior_draft.strip():
        base_sentences = [prior_draft.strip()]
    return base_sentences, extra_tags


def _apply_style_constraints_and_validate(
    *,
    sentences: list[str],
    raw_source: str,
    profile: WordingPreferenceProfile,
    extra_tags: list[str] | None = None,
    preserve_live_formatting: bool = False,
) -> tuple[str, int]:
    """Apply user wording preferences and run self-healing tone validation."""
    raw_tags = _HASHTAG_RE.findall(raw_source)
    candidate_tags = list(dict.fromkeys([*raw_tags, *(extra_tags or [])]))
    kept_tags = candidate_tags[:5]

    cleaned_sentences: list[str] = []
    for sentence in sentences:
        text = sentence
        for banned in profile.banned_words:
            pat = re.compile(rf"\b{re.escape(banned)}\w*\b", re.IGNORECASE)
            text = pat.sub("focused", text)
        if profile.avoid_emojis:
            text = _EMOJI_RE.sub("", text)
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        if text:
            cleaned_sentences.append(text)

    if not cleaned_sentences:
        cleaned_sentences = [
            "Looking ahead to meaningful engineering challenges and continuous growth."
        ]

    if preserve_live_formatting:
        draft = "\n\n".join(cleaned_sentences)
    elif profile.preferred_format == PreferredFormat.BULLETS:
        bullets = list(cleaned_sentences)
        if len(bullets) < 2:
            bullets.append(
                "Focused on practical execution, measurable impact, and continuous growth."
            )
        draft = "\n".join(f"- {item}" for item in bullets)
    elif profile.preferred_format == PreferredFormat.BROETRY:
        draft = "\n\n".join(cleaned_sentences)
    elif profile.keep_concise:
        draft = cleaned_sentences[0]
    else:
        draft = " ".join(cleaned_sentences)

    if kept_tags:
        clean_tags = [
            _EMOJI_RE.sub("", t) if profile.avoid_emojis else t for t in kept_tags
        ]
        existing_tags = _HASHTAG_RE.findall(draft)
        if not preserve_live_formatting or not existing_tags:
            missing_tags = [t for t in clean_tags if len(t) > 1 and t.lower() not in draft.lower()]
            tag_str = " ".join(missing_tags[: max(0, 5 - len(existing_tags))])
            if tag_str:
                draft = f"{draft}\n\n{tag_str}"

    if len(draft) > 3000:
        draft = draft[:2990].rstrip() + "..."

    validation = score_tone_and_bridge_burn_risk(
        candidate_text=draft,
        target_style="LINKEDIN_SPEAK",
        banned_words=profile.banned_words,
        avoid_emojis=profile.avoid_emojis,
        max_characters=3000,
    )
    if validation["status"] == "recoverable_error":
        for banned in profile.banned_words:
            draft = re.sub(
                rf"\b{re.escape(banned)}\w*\b",
                "energized",
                draft,
                flags=re.IGNORECASE,
            )
        if profile.avoid_emojis:
            draft = _EMOJI_RE.sub("", draft)
        if len(draft) > 3000:
            draft = draft[:2990].rstrip() + "..."
        validation = score_tone_and_bridge_burn_risk(
            candidate_text=draft,
            target_style="LINKEDIN_SPEAK",
            banned_words=profile.banned_words,
            avoid_emojis=profile.avoid_emojis,
            max_characters=3000,
        )

    risk_score = int(validation.get("bridge_burn_risk_score", 8))
    return draft.strip(), risk_score


def create_app() -> FastAPI:
    """Construct the FastAPI conversational application bound to live settings."""
    settings = get_settings()
    if settings.google_genai_use_vertexai:
        for key_var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
            val = os.getenv(key_var, "").strip().lower()
            if val in {"", "none", "placeholder", "replace-me", "changeme"}:
                os.environ.pop(key_var, None)
    configure_logging(
        project_id=settings.google_cloud_project,
    )
    configure_tracing(
        enable_cloud_export=settings.otel_export_to_cloud,
        project_id=settings.google_cloud_project,
    )

    scrubber = get_default_scrubber()
    style_store = get_default_style_store(settings.data_dir)
    session_service = create_session_service(data_dir=settings.data_dir)
    memory_service = create_memory_service(
        style_store=style_store,
        data_dir=settings.data_dir,
    )
    adk_app = create_adk_app()
    adk_runner = Runner(
        app=adk_app,
        session_service=session_service,
        memory_service=memory_service,
        auto_create_session=True,
    )
    auth_mode = get_secret_resolver().resolve_auth_mode(settings=settings)
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"

    session_drafts: dict[str, str] = {}
    session_sources: dict[str, str] = {}

    async def _persist_adk_turn(
        *,
        session_id: str,
        user_id: str,
        user_message: str,
        assistant_reply: str,
    ) -> None:
        """Persist scrubbed conversational turn into ADK SqliteSessionService."""
        try:
            existing = await session_service.get_session(
                app_name=adk_app.name,
                user_id=user_id,
                session_id=session_id,
            )
            if existing is None:
                existing = await session_service.create_session(
                    app_name=adk_app.name,
                    user_id=user_id,
                    session_id=session_id,
                    state={"last_reply": assistant_reply[:500]},
                )
            else:
                existing.state["last_reply"] = assistant_reply[:500]
            from google.adk.events import Event

            user_evt = Event(
                author="user",
                content=types.Content(
                    role="user",
                    parts=[types.Part(text=user_message)],
                ),
            )
            model_evt = Event(
                author="coordinator_agent",
                content=types.Content(
                    role="model",
                    parts=[types.Part(text=assistant_reply)],
                ),
            )
            await session_service.append_event(existing, user_evt)
            await session_service.append_event(existing, model_evt)
        except Exception as exc:
            get_logger(__name__).warning(
                "session_turn_persist_warning",
                error_message=scrubber.scrub(str(exc)).scrubbed_text,
            )

    async def _maybe_run_live_adk(
        *,
        user_id: str,
        session_id: str,
        prompt_text: str,
    ) -> str | None:
        """Invoke the live multi-agent ADK Runner when cloud credentials exist."""
        if auth_mode == "unconfigured":
            return None
        if settings.google_genai_use_vertexai and not settings.google_cloud_project:
            return None

        try:
            preferred_final: str | None = None
            last_final: str | None = None
            async for event in adk_runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text=prompt_text)],
                ),
            ):
                if (
                    event.is_final_response()
                    and event.content
                    and event.content.parts
                ):
                    text_parts = [
                        part.text.strip()
                        for part in event.content.parts
                        if getattr(part, "text", None) and part.text.strip()
                    ]
                    if text_parts:
                        joined = "\n\n".join(text_parts)
                        last_final = joined
                        if getattr(event, "author", "") in {
                            "self_eval_auditor_agent",
                            "coordinator_agent",
                            "linkedin_converter_agent",
                        }:
                            preferred_final = joined
            return preferred_final or last_final
        except Exception as exc:
            get_logger(__name__).warning(
                "live_adk_runner_fallback",
                error_type=type(exc).__name__,
                error_message=scrubber.scrub(str(exc)).scrubbed_text[:240],
            )
            return None

    @asynccontextmanager
    async def _lifespan(_fastapi_app: FastAPI) -> AsyncIterator[None]:
        yield
        await drain_memory_tasks()

    fastapi_app = FastAPI(
        title="Thrilled to Announce",
        description="Clean LinkedIn-styled conversational Human <-> LinkedIn translator.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @fastapi_app.get("/", response_class=HTMLResponse)
    async def serve_index() -> HTMLResponse:
        """Serve the distraction-free single-column `ta` conversational UI."""
        index_file = frontend_dir / "index.html"
        return HTMLResponse(
            content=index_file.read_text(encoding="utf-8"),
            status_code=200,
        )

    @fastapi_app.get("/health")
    @fastapi_app.get("/healthz")
    async def healthz() -> dict[str, str]:
        """Health check endpoint for Cloud Run startup/liveness probes."""
        return {"status": "ok", "service": "thrilled-to-announce"}

    @fastapi_app.get("/api/welcome", response_model=WelcomeResponse)
    async def get_welcome() -> WelcomeResponse:
        """Return concise welcome greeting and two bidirectional starter prompts."""
        return _WELCOME_PAYLOAD

    @fastapi_app.post("/api/chat", response_model=ChatTurnResponse)
    async def post_chat(req: ChatTurnRequest) -> ChatTurnResponse:
        """Auto-detect conversational intent, convert or save style, and return reply."""
        if not req.message.strip():
            raise HTTPException(
                status_code=422,
                detail="Message cannot be empty or whitespace only.",
            )

        session_id = req.session_id or f"sess_{uuid.uuid4().hex[:14]}"
        user_id = req.user_id

        with IntentOutcomeContext(
            "http.post.api_chat",
            operation_kind="http",
            session_id=session_id,
            user_id=user_id,
            args={"message": req.message[:200]},
        ) as telemetry_ctx:
            decision = evaluate_input_policy(req.message, scrubber=scrubber)
            scrubbed_msg = decision.scrubbed_text
            pii_count = decision.pii_redactions_count

            if decision.blocked:
                refusal = decision.reply_text or (
                    "That looks like non-public company info. Remove unreleased "
                    "numbers or internal codenames and send it again!"
                )
                await _persist_adk_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message="[POLICY_BLOCKED_INPUT]",
                    assistant_reply=refusal,
                )
                telemetry_ctx.set_outcome(
                    status="blocked",
                    result={"guardrail_blocked": True, "reason": decision.reason},
                )
                return ChatTurnResponse(
                    session_id=session_id,
                    detected_style=DetectedStyle.HUMAN_DAILY,
                    label=decision.label,
                    reply_text=refusal,
                    preference_saved=False,
                    used_saved_preference=False,
                    can_post_to_linkedin=False,
                    hitl_required=False,
                    pending_publish_id=None,
                    guardrail_blocked=True,
                    pii_redactions_count=pii_count,
                    bridge_burn_risk_score=0,
                )

            has_prev = session_id in session_drafts
            with IntentOutcomeContext(
                "detect_human_or_linkedin_style",
                operation_kind="tool",
                session_id=session_id,
                user_id=user_id,
                args={"has_previous_draft": has_prev},
            ) as detect_ctx:
                style_res = detect_human_or_linkedin_style(
                    message=scrubbed_msg,
                    has_previous_draft=has_prev,
                )
                detect_ctx.set_outcome(
                    status=style_res["status"],
                    result={"detected_style": style_res["detected_style"]},
                )
            detected = DetectedStyle(style_res["detected_style"])

            pref_res = retrieve_wording_preferences(
                user_id=user_id,
                query_text=scrubbed_msg[:4000],
                top_k=3,
            )
            profile = WordingPreferenceProfile.model_validate(pref_res["profile"])

            if detected == DetectedStyle.PUBLISH_REQUEST:
                prior_draft = session_drafts.get(session_id, "")
                if not prior_draft:
                    msg = (
                        "Send me a thought to turn into a LinkedIn draft first, "
                        "and then I'll ask for your confirmation before posting."
                    )
                    telemetry_ctx.set_outcome(status="ok", result={"hitl": False})
                    return ChatTurnResponse(
                        session_id=session_id,
                        detected_style=DetectedStyle.PUBLISH_REQUEST,
                        label="Assistant",
                        reply_text=msg,
                        can_post_to_linkedin=False,
                        hitl_required=False,
                        pending_publish_id=None,
                        pii_redactions_count=pii_count,
                        bridge_burn_risk_score=0,
                    )

                approval = request_publish_approval(
                    post_content=prior_draft,
                    user_id=user_id,
                )
                pub_id = str(approval["publish_id"])
                telemetry_ctx.set_outcome(
                    status="awaiting_human_confirmation",
                    result={"publish_id": pub_id},
                )
                return ChatTurnResponse(
                    session_id=session_id,
                    detected_style=DetectedStyle.PUBLISH_REQUEST,
                    label="LinkedIn Style",
                    reply_text=prior_draft,
                    used_saved_preference=profile.has_custom_preferences,
                    can_post_to_linkedin=True,
                    hitl_required=True,
                    pending_publish_id=pub_id,
                    pii_redactions_count=pii_count,
                    bridge_burn_risk_score=8,
                )

            if detected == DetectedStyle.STYLE_TWEAK:
                with IntentOutcomeContext(
                    "save_wording_preference",
                    operation_kind="tool",
                    session_id=session_id,
                    user_id=user_id,
                    args={"instruction": scrubbed_msg[:180]},
                ) as save_ctx:
                    saved_res = save_wording_preference(
                        user_id=user_id,
                        raw_instruction=scrubbed_msg,
                    )
                    save_ctx.set_outcome(
                        status=saved_res["status"],
                        result={
                            "persistence_mode": saved_res["persistence_mode"]
                        },
                    )

                updated_profile = WordingPreferenceProfile.model_validate(
                    saved_res["updated_profile"]
                )
                prior_draft = session_drafts.get(session_id, "")
                source_text = session_sources.get(
                    session_id,
                    "I am tired of my current company and just wanted to find something not that boring",
                )
                live_reply = await _maybe_run_live_adk(
                    user_id=user_id,
                    session_id=session_id,
                    prompt_text=(
                        f"Apply this modification or style preference directly to the current draft: {scrubbed_msg}\n"
                        f"Current draft: {prior_draft or '(none yet)'}\n"
                        f"Original human thought: {source_text}\n"
                        f"Active profile: {updated_profile.model_dump_json()}\n"
                        "Return ONLY the revised LinkedIn post (no commentary)."
                    ),
                )
                if live_reply:
                    scrubbed_live = scrubber.scrub(live_reply).scrubbed_text
                    _, extra_tags = _apply_followup_tweak_to_sentences(
                        sentences=[scrubbed_live],
                        prior_draft=prior_draft,
                        source_text=source_text,
                        tweak_instruction=scrubbed_msg,
                    )
                    rewritten, risk_score = _apply_style_constraints_and_validate(
                        sentences=[scrubbed_live],
                        raw_source=source_text,
                        profile=updated_profile,
                        extra_tags=extra_tags,
                        preserve_live_formatting=True,
                    )
                else:
                    sentences = _compose_linkedin_sentences(source_text)
                    tweaked_sentences, extra_tags = _apply_followup_tweak_to_sentences(
                        sentences=sentences,
                        prior_draft=prior_draft,
                        source_text=source_text,
                        tweak_instruction=scrubbed_msg,
                    )
                    rewritten, risk_score = _apply_style_constraints_and_validate(
                        sentences=tweaked_sentences,
                        raw_source=source_text,
                        profile=updated_profile,
                        extra_tags=extra_tags,
                    )

                session_drafts[session_id] = rewritten
                session_sources[session_id] = source_text
                await _persist_adk_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=scrubbed_msg,
                    assistant_reply=rewritten,
                )
                telemetry_ctx.set_outcome(
                    status="ok",
                    result={"detected_style": "STYLE_TWEAK", "preference_saved": True},
                )
                return ChatTurnResponse(
                    session_id=session_id,
                    detected_style=DetectedStyle.STYLE_TWEAK,
                    label="LinkedIn Style",
                    reply_text=rewritten,
                    preference_saved=True,
                    used_saved_preference=True,
                    can_post_to_linkedin=True,
                    hitl_required=False,
                    pending_publish_id=None,
                    guardrail_blocked=False,
                    pii_redactions_count=pii_count,
                    bridge_burn_risk_score=risk_score,
                )

            if detected == DetectedStyle.LINKEDIN_SPEAK:
                live_decoded = await _maybe_run_live_adk(
                    user_id=user_id,
                    session_id=session_id,
                    prompt_text=(
                        "Decode this LinkedIn corporate post into 1-2 candid, "
                        f"plain English sentences:\n{scrubbed_msg}"
                    ),
                )
                decoded = (
                    scrubber.scrub(live_decoded).scrubbed_text
                    if live_decoded
                    else _decode_linkedin_subtext(scrubbed_msg)
                )
                await _persist_adk_turn(
                    session_id=session_id,
                    user_id=user_id,
                    user_message=scrubbed_msg,
                    assistant_reply=decoded,
                )
                telemetry_ctx.set_outcome(
                    status="ok",
                    result={"detected_style": "LINKEDIN_SPEAK"},
                )
                return ChatTurnResponse(
                    session_id=session_id,
                    detected_style=DetectedStyle.LINKEDIN_SPEAK,
                    label="Plain English",
                    reply_text=decoded,
                    preference_saved=False,
                    used_saved_preference=False,
                    can_post_to_linkedin=False,
                    hitl_required=False,
                    pending_publish_id=None,
                    guardrail_blocked=False,
                    pii_redactions_count=pii_count,
                    bridge_burn_risk_score=4,
                )

            # HUMAN_DAILY -> LINKEDIN_SPEAK
            live_converted = await _maybe_run_live_adk(
                user_id=user_id,
                session_id=session_id,
                prompt_text=(
                    "Convert this plain human thought into witty, career-safe LinkedIn style "
                    "while keeping an unmistakable thematic connection so both sides feel clearly "
                    "linked and funny.\n"
                    f"Input: {scrubbed_msg}\n"
                    f"Saved preferences: {profile.model_dump_json()}\n"
                    "Return ONLY the LinkedIn post."
                ),
            )
            if live_converted:
                scrubbed_live = scrubber.scrub(live_converted).scrubbed_text
                draft, risk_score = _apply_style_constraints_and_validate(
                    sentences=[scrubbed_live],
                    raw_source=scrubbed_msg,
                    profile=profile,
                    preserve_live_formatting=True,
                )
            else:
                sentences = _compose_linkedin_sentences(scrubbed_msg)
                draft, risk_score = _apply_style_constraints_and_validate(
                    sentences=sentences,
                    raw_source=scrubbed_msg,
                    profile=profile,
                )

            session_sources[session_id] = scrubbed_msg
            session_drafts[session_id] = draft
            await _persist_adk_turn(
                session_id=session_id,
                user_id=user_id,
                user_message=scrubbed_msg,
                assistant_reply=draft,
            )
            telemetry_ctx.set_outcome(
                status="ok",
                result={
                    "detected_style": "HUMAN_DAILY",
                    "used_saved_preference": profile.has_custom_preferences,
                    "bridge_burn_risk_score": risk_score,
                },
            )
            return ChatTurnResponse(
                session_id=session_id,
                detected_style=DetectedStyle.HUMAN_DAILY,
                label="LinkedIn Style",
                reply_text=draft,
                preference_saved=False,
                used_saved_preference=profile.has_custom_preferences,
                can_post_to_linkedin=True,
                hitl_required=False,
                pending_publish_id=None,
                guardrail_blocked=False,
                pii_redactions_count=pii_count,
                bridge_burn_risk_score=risk_score,
            )

    @fastapi_app.post("/api/publish/request", response_model=PublishFeedResponse)
    async def post_publish_request(req: PublishFeedRequest) -> PublishFeedResponse:
        """Stage a HITL confirmation gate prior to LinkedIn posting."""
        with IntentOutcomeContext(
            "http.post.publish_request",
            operation_kind="http",
            session_id="publish_gate",
            user_id=req.user_id,
            args={"post_length": len(req.post_content)},
        ) as ctx:
            decision = evaluate_input_policy(req.post_content, scrubber=scrubber)
            if decision.blocked:
                ctx.set_outcome(status="blocked", result={"reason": decision.reason})
                return PublishFeedResponse(
                    status="recoverable_error",
                    publish_id="blocked-by-policy",
                    preview_text="",
                    message=decision.reply_text or "Blocked by policy.",
                    recovery_instructions=[
                        "Remove non-public or harmful content before publishing."
                    ],
                )

            res = request_publish_approval(
                post_content=decision.scrubbed_text,
                user_id=req.user_id,
            )
            ctx.set_outcome(status=res["status"], result={"publish_id": res["publish_id"]})
            return PublishFeedResponse.model_validate(res)

    @fastapi_app.post("/api/publish/confirm", response_model=PublishFeedResponse)
    async def post_publish_confirm(
        req: PublishConfirmHttpRequest,
    ) -> PublishFeedResponse:
        """Complete or cancel a pending HITL publish confirmation."""
        with IntentOutcomeContext(
            "http.post.publish_confirm",
            operation_kind="http",
            session_id="publish_gate",
            user_id="hitl_reviewer",
            args={"publish_id": req.publish_id, "confirmed": req.confirmed},
        ) as ctx:
            res = confirm_publish_decision(
                publish_id=req.publish_id,
                confirmed=req.confirmed,
            )
            ctx.set_outcome(status=res["status"], result={"publish_id": res["publish_id"]})
            return PublishFeedResponse.model_validate(res)

    @fastapi_app.get("/api/telemetry")
    async def get_telemetry() -> dict[str, list[dict[str, Any]]]:
        """Return scrubbed structured JSON logs and OpenTelemetry spans."""
        return {
            "logs": get_telemetry_logs(limit=200),
            "spans": get_recorded_spans(limit=200),
        }

    if frontend_dir.exists():
        fastapi_app.mount(
            "/static",
            StaticFiles(directory=str(frontend_dir)),
            name="static",
        )

    return fastapi_app


app = create_app()
