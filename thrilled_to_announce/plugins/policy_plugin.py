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
"""ADK safety and confidentiality policy plugin blocking NDA leaks, doxxing, and prompt injection."""

from __future__ import annotations

import re
from typing import Any

from google.adk.agents.base_agent import BaseAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.api_core.client_options import ClientOptions
from google.genai import types
from google.genai.types import HarmBlockThreshold, HarmCategory, SafetySetting
from pydantic import BaseModel, ConfigDict, Field

from thrilled_to_announce.config.settings import AppSettings, get_settings
from thrilled_to_announce.observability.logging import get_logger
from thrilled_to_announce.observability.pii_scrubber import (
    PiiScrubber,
    get_default_scrubber,
)

NDA_REFUSAL_TEXT = (
    "That looks like non-public company info. Remove unreleased numbers or "
    "internal codenames and send it again!"
)
DOXXING_REFUSAL_TEXT = (
    "I cannot help share private personal addresses, personal contact details, "
    "or targeted harassment. Rephrase around your own career update and send it again!"
)
PROMPT_INJECTION_REFUSAL_TEXT = (
    "I can only convert between candid work thoughts and LinkedIn posts or save "
    "your wording preferences. Send a workplace thought or LinkedIn post to continue!"
)

_POLICY_BLOCKED_PLACEHOLDER = "[POLICY_BLOCKED_INPUT]"

_NDA_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:unreleased|unannounced|pre-announcement|embargoed|non-public|"
        r"confiden[t]ial|internal[- ]only|leaked)\b[^.!?\n]{0,90}\b"
        r"(?:q[1-4]|fy\d{2,4}|revenue|arr|mrr|earnings|ebitda|margin|guidance|"
        r"numbers|metrics|financials|valuation|acquisition|merger|m&a|"
        r"layoff\s+list|roadmap|codename|project)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:q[1-4]\s+revenue|quarterly\s+revenue|earnings\s+report|"
        r"acquisition\s+target|m&a\s+deal|internal\s+metrics)\b[^.!?\n]{0,90}\b"
        r"(?:unreleased|unannounced|embargoed|non-public|under\s+nda|"
        r"confiden[t]ial|before\s+(?:earnings|announcement))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:(?:\bour|\bmy\s+company'?s|\binternal)\s+(?:unreleased\s+)?"
        r"(?:q[1-4]\s+)?(?:revenue|arr|earnings|ebitda)\s*"
        r"(?:of\s+|is\s+|was\s+|hit\s+|\()?\s*\$\d+(?:\.\d+)?\s*"
        r"(?:[mbt]\b|mm\b|bn\b|mn\b|million\b|billion\b)"
        r"|\$\d+(?:\.\d+)?\s*(?:[mbt]\b|mm\b|bn\b|mn\b|million\b|billion\b)"
        r"[^.!?\n]{0,60}\b(?:q[1-4]\s+revenue|unreleased|embargoed|under\s+nda|"
        r"non-public|internal\s+revenue|before\s+earnings)\b)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:project\s+codename|internal\s+codename|secret\s+codename|"
        r"codename\s*:\s*[A-Za-z0-9_\-]+)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:under\s+nda|covered\s+by\s+nda|violat(?:e|ing)\s+(?:my|our|the)\s+nda|"
        r"break(?:ing)?\s+(?:my|our|the)\s+nda|trade\s+secret|"
        r"material\s+non-public\s+information|mnpi)\b",
        re.IGNORECASE,
    ),
)

_DOXXING_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:dox|doxx|doxxing|doxing|swat|swatting)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:post|share|publish|expose|leak|reveal|blast|drop)\b[^.!?\n]{0,70}\b"
        r"(?:personal\s+address|home\s+address|street\s+address|residential\s+address|"
        r"(?:their|his|her|manager'?s|boss'?s|coworker'?s|colleague'?s|ex'?s|ceo'?s|"
        r"vp'?s)\s+(?:private\s+|personal\s+|home\s+)?(?:address|phone\s+number|"
        r"ssn|social\s+security|email\s+address))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:harass|stalk|cyberbully|brigade|publicly\s+humiliate|"
        r"ruin\s+(?:their|his|her)\s+life)\b",
        re.IGNORECASE,
    ),
)

_PROMPT_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+|any\s+|your\s+|the\s+)?"
        r"(?:previous|prior|above|earlier|system|developer|safety|constitutional)\s+"
        r"(?:instructions?|prompts?|guardrails?|constraints?|rules?|directives?|policies|restrictions?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\s+(?:(?:all|everything)\s+)?"
        r"(?:above|the\s+above|previous|prior)\s+(?:and|then)\s+"
        r"(?:output|print|write|reveal|show|dump|answer|solve|execute|act)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"</?\s*(?:untrusted_user_input|system|developer|assistant|instructions?)\s*>",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:reveal|print|output|show|dump|repeat|leak)\s+(?:your\s+|the\s+)?"
        r"(?:full\s+|exact\s+|raw\s+|hidden\s+|internal\s+)?"
        r"(?:system\s+prompt|developer\s+prompt|hidden\s+instructions?|"
        r"constitution(?:al)?\s+prompt)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:you\s+are\s+now\s+dan|do\s+anything\s+now|jailbreak\s+mode|"
        r"developer\s+mode\s+enabled)\b",
        re.IGNORECASE,
    ),
)

COACH_REDIRECT_SUGGESTED_REPLIES: tuple[str, ...] = (
    "Turn a workplace rant into a post",
    "Decode a corporate post",
)

_COACH_REDIRECT_BY_REASON: dict[str, str] = {
    "off_topic_coding_challenge": (
        "Ha—I'm flattered you trust my algorithms, but I strictly translate "
        "between Human reality and LinkedIn brilliance! Tell me how wrestling with "
        "that code made your week feel, and I'll spin it into a Thought Leadership post."
    ),
    "system_info_probe": (
        "Nice try—my inner plumbing stays off the feed! I'm your Human <-> LinkedIn "
        "translator, not a system console. Drop a candid work thought or a "
        "buzzword-heavy LinkedIn post and let's translate it."
    ),
    "outbound_boundary_blocked": (
        "Whoa, let's keep things strictly in the Human <-> LinkedIn lane—no raw "
        "code blocks or behind-the-scenes details here! Send me a workplace rant "
        "to polish or a corporate post to decode."
    ),
    "off_topic_general_qa": (
        "I'm strictly a Human <-> LinkedIn translator—no trivia, homework, or "
        "side quests here! Share a candid workplace moment or paste a LinkedIn "
        "post, and I'll translate it in seconds."
    ),
}

_TEMPLATE_LOCATION_RE = re.compile(
    r"^projects/[^/]+/locations/([^/]+)/templates/[^/]+$"
)

_OUTBOUND_LEAK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bgemini-\d[\w.-]*\b", re.IGNORECASE),
    re.compile(r"\bvertexai\b", re.IGNORECASE),
    re.compile(
        r"\b(?:GLOBAL_CONSTITUTION|COORDINATOR_INSTRUCTION|"
        r"SUBTEXT_ANALYST_INSTRUCTION|TRANSLATOR_INSTRUCTION|"
        r"REACTION_SIMULATOR_INSTRUCTION|SELF_EVAL_AUDITOR_INSTRUCTION|"
        r"BridgeBurnAndNdaPolicyPlugin|IntentOutcomeTelemetryPlugin|"
        r"EventsCompactionConfig|ReflectAndRetryToolPlugin)\b"
    ),
    re.compile(
        r"\b(?:GOOGLE_CLOUD_PROJECT|GOOGLE_CLOUD_LOCATION|GEMINI_API_KEY|"
        r"GEMINI_API_KEY_SECRET_ID|GOOGLE_GENAI_USE_VERTEXAI|"
        r"THRILLED_FLASH_MODEL|THRILLED_PRO_MODEL|THRILLED_DATA_DIR|"
        r"ENABLE_CLOUD_DLP|OTEL_EXPORT_TO_CLOUD)\b"
    ),
    re.compile(r"\bthrilled-ta-[a-z0-9_-]*", re.IGNORECASE),
    re.compile(r"```"),
    re.compile(
        r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:"
    ),
    re.compile(
        r"(?m)^\s*class\s+[A-Z]\w*(?:\s*\([^)]*\)|\s+extends\s+\w+)?\s*[:{]"
    ),
    re.compile(
        r"(?m)^\s*(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{"
    ),
    re.compile(
        r"(?is)\bSELECT\b[\s\S]{1,120}\bFROM\b[\s\S]{0,120}"
        r"(?:\bWHERE\b|\bJOIN\b|\bGROUP\s+BY\b|\bORDER\s+BY\b|;|\n)"
    ),
)


class PolicyDecision(BaseModel):
    """Result of evaluating user or model/tool text against safety and NDA policies."""

    model_config = ConfigDict(extra="forbid")

    blocked: bool = False
    reason: str | None = None
    label: str = "OK"
    reply_text: str | None = None
    scrubbed_text: str
    pii_redactions_count: int = Field(default=0, ge=0)


def default_safety_settings() -> list[SafetySetting]:
    """Return Gemini content safety settings enforced across all LLM requests."""
    return [
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
        SafetySetting(
            category=HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        ),
    ]


def _match_policy_violation(text: str) -> tuple[str, str] | None:
    """Check candidate text against NDA, anti-doxxing, and prompt-injection rules."""
    if not text or text == _POLICY_BLOCKED_PLACEHOLDER:
        return None
    for pattern in _NDA_PATTERNS:
        if pattern.search(text):
            return ("nda_or_non_public_corporate_info", NDA_REFUSAL_TEXT)
    for pattern in _DOXXING_PATTERNS:
        if pattern.search(text):
            return ("doxxing_or_targeted_harassment", DOXXING_REFUSAL_TEXT)
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return ("prompt_injection_attempt", PROMPT_INJECTION_REFUSAL_TEXT)
    return None


def evaluate_input_policy(
    text: str,
    scrubber: PiiScrubber | None = None,
) -> PolicyDecision:
    """Evaluate text for NDA leaks, doxxing, and prompt injection while scrubbing PII."""
    active_scrubber = scrubber or get_default_scrubber()
    scrub_res = active_scrubber.scrub(text or "")

    violation = _match_policy_violation(text) or _match_policy_violation(
        scrub_res.scrubbed_text
    )
    if violation is not None:
        reason, refusal_text = violation
        return PolicyDecision(
            blocked=True,
            reason=reason,
            label="Confidentiality Check",
            reply_text=refusal_text,
            scrubbed_text=scrub_res.scrubbed_text,
            pii_redactions_count=scrub_res.redactions_count,
        )

    return PolicyDecision(
        blocked=False,
        reason=None,
        label="OK",
        reply_text=None,
        scrubbed_text=scrub_res.scrubbed_text,
        pii_redactions_count=scrub_res.redactions_count,
    )


def off_topic_coach_redirect(
    reason: str | None = None,
) -> tuple[str, str, list[str]]:
    """Return a witty Coach redirect, follow-up prompt, and quick-reply chips."""
    reply_text = _COACH_REDIRECT_BY_REASON.get(
        reason or "",
        _COACH_REDIRECT_BY_REASON["off_topic_general_qa"],
    )
    proactive_prompt = (
        "Got a real work frustration to turn into a post, or a corporate update "
        "you want me to decode?"
    )
    return reply_text, proactive_prompt, list(COACH_REDIRECT_SUGGESTED_REPLIES)


def evaluate_topic_scope(
    text: str,
    *,
    has_active_draft: bool = False,
) -> PolicyDecision:
    """Non-regex topic scope pass-through; live screening uses Model Armor + Gemini scope."""
    del has_active_draft
    scrub_res = get_default_scrubber().scrub(text or "")
    return PolicyDecision(
        blocked=False,
        reason=None,
        label="OK",
        reply_text=None,
        scrubbed_text=scrub_res.scrubbed_text,
        pii_redactions_count=scrub_res.redactions_count,
    )


def _extract_template_location(template_name: str | None) -> str | None:
    if not template_name:
        return None
    match = _TEMPLATE_LOCATION_RE.match(template_name.strip())
    return match.group(1) if match else None


def _build_model_armor_async_client(location: str) -> Any:
    from google.cloud import modelarmor_v1

    return modelarmor_v1.ModelArmorAsyncClient(
        client_options=ClientOptions(
            api_endpoint=f"modelarmor.{location}.rep.googleapis.com"
        )
    )


def create_model_armor_plugin(
    settings: AppSettings | None = None,
    *,
    client: Any = None,
) -> BasePlugin | None:
    """Create ADK's native ``ModelArmorPlugin`` when Model Armor templates are configured."""
    cfg = settings or get_settings()
    if not cfg.enable_model_armor:
        return None
    prompt_tpl = cfg.model_armor_prompt_template
    response_tpl = cfg.model_armor_response_template
    if not prompt_tpl and not response_tpl:
        return None

    try:
        from google.adk.integrations.model_armor import (
            ModelArmorConfig,
            ModelArmorPlugin,
        )
    except ImportError as exc:
        get_logger(__name__).warning(
            "model_armor_plugin_unavailable",
            error_type=type(exc).__name__,
            error_message=str(exc)[:200],
        )
        return None

    outbound_redirect, _, _ = off_topic_coach_redirect("outbound_boundary_blocked")
    plugin_cfg = ModelArmorConfig(
        prompt_template_name=prompt_tpl,
        response_template_name=response_tpl,
        input_blocked_message=PROMPT_INJECTION_REFUSAL_TEXT,
        output_blocked_message=outbound_redirect,
        block_on_screening_failure=False,
    )
    return ModelArmorPlugin(config=plugin_cfg, client=client)


async def screen_user_prompt_with_model_armor(
    text: str,
    *,
    settings: AppSettings | None = None,
    client: Any = None,
    scrubber: PiiScrubber | None = None,
) -> PolicyDecision:
    """Screen inbound user text with Google Cloud Model Armor (prompt injection/jailbreak)."""
    active_scrubber = scrubber or get_default_scrubber()
    scrub_res = active_scrubber.scrub(text or "")
    cfg = settings or get_settings()

    if not cfg.enable_model_armor or not cfg.model_armor_prompt_template:
        return PolicyDecision(
            blocked=False,
            reason=None,
            label="OK",
            reply_text=None,
            scrubbed_text=scrub_res.scrubbed_text,
            pii_redactions_count=scrub_res.redactions_count,
        )

    template_name = cfg.model_armor_prompt_template
    location = _extract_template_location(template_name) or cfg.google_cloud_location
    try:
        from google.cloud import modelarmor_v1

        ma_client = client or _build_model_armor_async_client(location)
        request = modelarmor_v1.SanitizeUserPromptRequest(
            name=template_name,
            user_prompt_data=modelarmor_v1.DataItem(text=text or ""),
        )
        response = await ma_client.sanitize_user_prompt(request=request)
        result = response.sanitization_result
        if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            filter_keys = sorted(getattr(result, "filter_results", {}).keys())
            return PolicyDecision(
                blocked=True,
                reason=(
                    "model_armor_prompt_injection"
                    if "pi_and_jailbreak" in filter_keys
                    else "model_armor_prompt_blocked"
                ),
                label="Confidentiality Check",
                reply_text=PROMPT_INJECTION_REFUSAL_TEXT,
                scrubbed_text=scrub_res.scrubbed_text,
                pii_redactions_count=scrub_res.redactions_count,
            )
    except Exception as exc:
        get_logger(__name__).warning(
            "model_armor_prompt_screen_fallback",
            error_type=type(exc).__name__,
            error_message=active_scrubber.scrub(str(exc)).scrubbed_text[:200],
        )

    return PolicyDecision(
        blocked=False,
        reason=None,
        label="OK",
        reply_text=None,
        scrubbed_text=scrub_res.scrubbed_text,
        pii_redactions_count=scrub_res.redactions_count,
    )


async def screen_model_response_with_model_armor(
    text: str,
    *,
    settings: AppSettings | None = None,
    client: Any = None,
    scrubber: PiiScrubber | None = None,
) -> PolicyDecision:
    """Screen outbound assistant text with Google Cloud Model Armor response template."""
    active_scrubber = scrubber or get_default_scrubber()
    scrub_res = active_scrubber.scrub(text or "")
    cfg = settings or get_settings()

    if (
        not text
        or not cfg.enable_model_armor
        or not cfg.model_armor_response_template
    ):
        return PolicyDecision(
            blocked=False,
            reason=None,
            label="OK",
            reply_text=None,
            scrubbed_text=scrub_res.scrubbed_text,
            pii_redactions_count=scrub_res.redactions_count,
        )

    template_name = cfg.model_armor_response_template
    location = _extract_template_location(template_name) or cfg.google_cloud_location
    try:
        from google.cloud import modelarmor_v1

        ma_client = client or _build_model_armor_async_client(location)
        request = modelarmor_v1.SanitizeModelResponseRequest(
            name=template_name,
            model_response_data=modelarmor_v1.DataItem(text=text),
        )
        response = await ma_client.sanitize_model_response(request=request)
        result = response.sanitization_result
        if result.filter_match_state == modelarmor_v1.FilterMatchState.MATCH_FOUND:
            redirect_text, _, _ = off_topic_coach_redirect("outbound_boundary_blocked")
            return PolicyDecision(
                blocked=True,
                reason="model_armor_response_blocked",
                label="Coach",
                reply_text=redirect_text,
                scrubbed_text=scrub_res.scrubbed_text,
                pii_redactions_count=scrub_res.redactions_count,
            )
    except Exception as exc:
        get_logger(__name__).warning(
            "model_armor_response_screen_fallback",
            error_type=type(exc).__name__,
            error_message=active_scrubber.scrub(str(exc)).scrubbed_text[:200],
        )

    return PolicyDecision(
        blocked=False,
        reason=None,
        label="OK",
        reply_text=None,
        scrubbed_text=scrub_res.scrubbed_text,
        pii_redactions_count=scrub_res.redactions_count,
    )


def evaluate_output_policy(
    coach_intro: str,
    post_text: str,
    scrubber: PiiScrubber | None = None,
) -> tuple[bool, str, str]:
    """Deterministic outbound check blocking leaked system identifiers or code dumps."""
    active_scrubber = scrubber or get_default_scrubber()
    raw_intro = coach_intro or ""
    raw_post = post_text or ""
    clean_intro = active_scrubber.scrub(raw_intro).scrubbed_text
    clean_post = active_scrubber.scrub(raw_post).scrubbed_text

    for surface in (f"{raw_intro}\n{raw_post}", f"{clean_intro}\n{clean_post}"):
        for pattern in _OUTBOUND_LEAK_PATTERNS:
            if pattern.search(surface):
                redirect_text, _, _ = off_topic_coach_redirect(
                    "outbound_boundary_blocked"
                )
                return False, "", redirect_text

    return True, clean_intro, clean_post


class BridgeBurnAndNdaPolicyPlugin(BasePlugin):
    """Enforces NDA, anti-doxxing, prompt-injection, and active PII scrubbing guardrails."""

    def __init__(
        self,
        name: str = "bridge_burn_and_nda_policy_plugin",
        scrubber: PiiScrubber | None = None,
    ) -> None:
        super().__init__(name=name)
        self._scrubber = scrubber
        self._invocation_decisions: dict[str, PolicyDecision] = {}
        self._flushed_redaction_invocations: set[str] = set()

    def _active_scrubber(self) -> PiiScrubber:
        return self._scrubber or get_default_scrubber()

    def _cleanup_invocation(self, invocation_id: str) -> None:
        self._invocation_decisions.pop(invocation_id, None)
        self._flushed_redaction_invocations.discard(invocation_id)

    def _sync_state_delta(self, invocation_id: str, state: Any) -> PolicyDecision | None:
        """Persist buffered ``on_user_message_callback`` policy metadata via ``state_delta``."""
        decision = self._invocation_decisions.get(invocation_id)
        if decision is None:
            return None

        if decision.blocked:
            state["guardrail_blocked"] = True
            state["guardrail_label"] = decision.label
            state["guardrail_reason"] = decision.reason
            state["guardrail_reply_text"] = decision.reply_text
        elif "guardrail_blocked" not in state:
            state["guardrail_blocked"] = False

        if invocation_id not in self._flushed_redaction_invocations:
            existing = int(state.get("pii_redactions_count", 0) or 0)
            state["pii_redactions_count"] = existing + decision.pii_redactions_count
            self._flushed_redaction_invocations.add(invocation_id)

        return decision

    async def on_user_message_callback(
        self,
        *,
        invocation_context: InvocationContext,
        user_message: types.Content,
    ) -> types.Content | None:
        """Inspect and scrub incoming user message before ADK appends it to session history."""
        if not user_message or not user_message.parts:
            return None

        scrubber = self._active_scrubber()
        new_parts: list[types.Part] = []
        mutated = False
        total_redactions = 0
        blocked_decision: PolicyDecision | None = None

        for part in user_message.parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text:
                decision = evaluate_input_policy(part_text, scrubber=scrubber)
                total_redactions += decision.pii_redactions_count
                if decision.blocked:
                    if blocked_decision is None:
                        blocked_decision = decision
                    new_parts.append(types.Part.from_text(text=_POLICY_BLOCKED_PLACEHOLDER))
                    mutated = True
                elif decision.scrubbed_text != part_text:
                    new_parts.append(types.Part.from_text(text=decision.scrubbed_text))
                    mutated = True
                else:
                    new_parts.append(part)
            else:
                new_parts.append(part)

        aggregate_decision = (
            PolicyDecision(
                blocked=True,
                reason=blocked_decision.reason,
                label=blocked_decision.label,
                reply_text=blocked_decision.reply_text,
                scrubbed_text=_POLICY_BLOCKED_PLACEHOLDER,
                pii_redactions_count=total_redactions,
            )
            if blocked_decision is not None
            else PolicyDecision(
                blocked=False,
                reason=None,
                label="OK",
                reply_text=None,
                scrubbed_text="",
                pii_redactions_count=total_redactions,
            )
        )
        self._invocation_decisions[invocation_context.invocation_id] = aggregate_decision

        get_logger(__name__).info(
            "policy_user_message_checked",
            session_id=invocation_context.session.id,
            user_id=invocation_context.user_id,
            guardrail_blocked=aggregate_decision.blocked,
            reason=aggregate_decision.reason,
            pii_redactions_count=total_redactions,
        )

        if mutated:
            return types.Content(
                role=user_message.role or "user",
                parts=new_parts,
            )
        return None

    async def before_agent_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
    ) -> types.Content | None:
        """Flush buffered policy state into ``callback_context.state`` so SQLite persists deltas."""
        del agent
        self._sync_state_delta(callback_context.invocation_id, callback_context.state)
        return None

    async def before_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
    ) -> LlmResponse | None:
        """Enforce safety settings, scrub PII in-place, and short-circuit blocked prompts."""
        if llm_request.config is not None and not llm_request.config.safety_settings:
            llm_request.config.safety_settings = default_safety_settings()

        buffered = self._sync_state_delta(
            callback_context.invocation_id, callback_context.state
        )
        scrubber = self._active_scrubber()
        additional_redactions = 0
        detected_block: PolicyDecision | None = (
            buffered if buffered is not None and buffered.blocked else None
        )

        for content in llm_request.contents or []:
            for part in content.parts or []:
                part_text = getattr(part, "text", None)
                if not isinstance(part_text, str) or not part_text:
                    continue
                decision = evaluate_input_policy(part_text, scrubber=scrubber)
                if decision.pii_redactions_count > 0:
                    additional_redactions += decision.pii_redactions_count
                if decision.scrubbed_text != part_text:
                    part.text = decision.scrubbed_text
                if decision.blocked and detected_block is None:
                    detected_block = decision

        if additional_redactions > 0:
            existing = int(callback_context.state.get("pii_redactions_count", 0) or 0)
            callback_context.state["pii_redactions_count"] = (
                existing + additional_redactions
            )
        elif "pii_redactions_count" not in callback_context.state:
            callback_context.state["pii_redactions_count"] = 0

        if detected_block is not None:
            refusal_text = detected_block.reply_text or NDA_REFUSAL_TEXT
            callback_context.state["guardrail_blocked"] = True
            callback_context.state["guardrail_label"] = detected_block.label
            callback_context.state["guardrail_reason"] = detected_block.reason
            callback_context.state["guardrail_reply_text"] = refusal_text
            get_logger(__name__).info(
                "policy_guardrail_blocked",
                session_id=callback_context.session.id,
                user_id=callback_context.user_id,
                agent_name=callback_context.agent_name,
                reason=detected_block.reason,
                label=detected_block.label,
            )
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=refusal_text)],
                ),
                turn_complete=True,
            )

        callback_context.state["guardrail_blocked"] = False
        return None

    def _inspect_and_scrub_tool_value(
        self,
        value: Any,
        scrubber: PiiScrubber,
    ) -> tuple[Any, int, PolicyDecision | None]:
        if isinstance(value, str):
            decision = evaluate_input_policy(value, scrubber=scrubber)
            return (
                decision.scrubbed_text,
                decision.pii_redactions_count,
                decision if decision.blocked else None,
            )
        if isinstance(value, dict):
            total_redactions = 0
            first_block: PolicyDecision | None = None
            for k in list(value.keys()):
                new_val, sub_cnt, sub_block = self._inspect_and_scrub_tool_value(
                    value[k], scrubber
                )
                value[k] = new_val
                total_redactions += sub_cnt
                if sub_block is not None and first_block is None:
                    first_block = sub_block
            return value, total_redactions, first_block
        if isinstance(value, list):
            total_redactions = 0
            first_block = None
            out_items: list[Any] = []
            for item in value:
                new_item, sub_cnt, sub_block = self._inspect_and_scrub_tool_value(
                    item, scrubber
                )
                out_items.append(new_item)
                total_redactions += sub_cnt
                if sub_block is not None and first_block is None:
                    first_block = sub_block
            return out_items, total_redactions, first_block
        return value, 0, None

    async def before_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
    ) -> dict[str, Any] | None:
        """Scrub PII inside ``tool_args`` in-place and block non-public tool inputs."""
        self._sync_state_delta(tool_context.invocation_id, tool_context.state)
        scrubber = self._active_scrubber()
        _, redactions_added, blocked_decision = self._inspect_and_scrub_tool_value(
            tool_args, scrubber
        )

        if redactions_added > 0:
            existing = int(tool_context.state.get("pii_redactions_count", 0) or 0)
            tool_context.state["pii_redactions_count"] = existing + redactions_added

        if blocked_decision is not None:
            refusal_text = blocked_decision.reply_text or NDA_REFUSAL_TEXT
            tool_context.state["guardrail_blocked"] = True
            tool_context.state["guardrail_label"] = blocked_decision.label
            tool_context.state["guardrail_reason"] = blocked_decision.reason
            get_logger(__name__).info(
                "policy_tool_call_blocked",
                session_id=tool_context.session.id,
                user_id=tool_context.user_id,
                tool_name=tool.name,
                reason=blocked_decision.reason,
                label=blocked_decision.label,
            )
            return {
                "status": "recoverable_error",
                "guardrail_blocked": True,
                "label": blocked_decision.label,
                "error_message": refusal_text,
                "recovery_instructions": [
                    "Please check tool inputs and never pass non-public numbers, codenames, or private addresses.",
                    "Present the Confidentiality Check refusal message directly without retrying non-public data.",
                ],
            }

        return None

    async def after_model_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_response: LlmResponse,
    ) -> LlmResponse | None:
        """Scrub any residual PII from model output parts before returning to the user."""
        if (
            llm_response.partial is True
            or llm_response.content is None
            or not llm_response.content.parts
        ):
            return None

        scrubber = self._active_scrubber()
        redactions_added = 0
        for part in llm_response.content.parts:
            part_text = getattr(part, "text", None)
            if not isinstance(part_text, str) or not part_text:
                continue
            scrubbed = scrubber.scrub(part_text)
            if scrubbed.redactions_count > 0:
                part.text = scrubbed.scrubbed_text
                redactions_added += scrubbed.redactions_count

        if redactions_added > 0:
            existing = int(callback_context.state.get("pii_redactions_count", 0) or 0)
            callback_context.state["pii_redactions_count"] = existing + redactions_added

        return None

    async def after_run_callback(
        self,
        *,
        invocation_context: InvocationContext,
    ) -> None:
        """Release per-invocation state buffers on normal run completion."""
        self._cleanup_invocation(invocation_context.invocation_id)

    async def on_run_error_callback(
        self,
        *,
        invocation_context: InvocationContext,
        error: Exception,
    ) -> None:
        """Release per-invocation state buffers on run failure."""
        del error
        self._cleanup_invocation(invocation_context.invocation_id)
