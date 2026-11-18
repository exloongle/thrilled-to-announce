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
"""Automatic 4-way conversational style and intent detection ADK tool."""

from __future__ import annotations

import re
from typing import Any

from thrilled_to_announce.memory.style_store import (
    get_structured_logger,
    scrub_pii_text,
)
from thrilled_to_announce.schemas.contracts import (
    DetectedStyle,
    DetectStyleRequest,
    DetectStyleResponse,
)

URL_SPAN_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)

PUBLISH_POS_RE = re.compile(
    r"(?:"
    r"\bpublish\b.*\b(?:linkedin|feed|post|draft|announcement|it|that|this|now)\b"
    r"|(?<!\ba\s)(?<!\ban\s)\b(?:post|share|push|send)(?:\s+(?:it|(?:this|that|the|my)(?:\s+(?:post|draft|announcement))?))?\s+(?:to|on)\s+(?:my\s+)?linkedin(?:\s+feed)?\b"
    r"|(?:^|[—\-.,!]\s*)(?:looks\s+great|perfect|love\s+it|ship\s+it|approved)[\s,—\-!]+(?:please\s+)?(?:publish|post)\b"
    r")",
    re.IGNORECASE,
)

PUBLISH_NEG_RE = re.compile(
    r"(?:"
    r"\b(?:turn|convert|translate|transform|rewrite|make)\b.*\binto\s+a\s+linkedin\b"
    r"|\b(?:doxx\w*|personal\s+address|home\s+address|harass\w*)\b"
    r"|\b(?:never\s+use|avoid|don'?t\s+use|do\s+not\s+use|save\s+(?:that|this)\s+(?:rule|preference))\b"
    r"|\b(?:don'?t|do\s+not|never|not|wait|hold\s+off|stop|should\s+i|can\s+i|whether\s+to)\b[^.?!]*\b(?:publish|post|share|push|send)\b"
    r")",
    re.IGNORECASE,
)

TWEAK_PERSIST_RE = re.compile(
    r"(?:"
    r"\b(?:save|remember|store|keep)\s+(?:this|that|my)\s+(?:style\s+)?(?:preference|rule|style|setting|for\s+next\s+time)\b"
    r"|\bremember\s+(?:this|that)\s+for\s+(?:next\s+time|future\s+posts?|all\s+my\s+posts?)\b"
    r"|\b(?:from\s+now\s+on|going\s+forward|for\s+future\s+(?:posts?|drafts?|conversions?))\b"
    r"|\b(?:switch|change|set|update|reset)\s+(?:my\s+)?(?:default\s+)?(?:style|tone|format|voice|preferences?)\s+(?:back\s+)?to\b"
    r")",
    re.IGNORECASE,
)

TWEAK_NEG_VOCAB_RE = re.compile(
    r"(?:"
    r"(?:^|[—\-;:,]\s*|\band\s+)(?:never|don'?t|do\s+not)\s+(?:use|include)\b"
    r"|\bavoid\s+(?:using\s+)?(?:the\s+(?:word|term|buzzword|phrase)s?|stiff\s+words?\s+like|buzzwords?\s+like|words?\s+like|phrases?\s+like|emojis?|hashtags?|bullet\s*points?)\b"
    r"|\bavoid\s+['\u2018\u2019\"\u201c\u201d][a-zA-Z]"
    r"|(?:^|[—\-;:,]\s*|\band\s+)avoid\s+[a-zA-Z]{3,25}(?:\s*$|[,.;!]|\s+in\s+my\s+posts?|\s+from\s+now\s+on)"
    r"|\b(?:ban|stop\s+using)\s+(?:the\s+)?(?:words?|buzzwords?|emojis?|hashtags?)\b"
    r"|\b(?:remove|drop|delete|cut|strip)\s+['\u2018\u2019\"\u201c\u201d][a-zA-Z]"
    r"|\b(?:you\s+can\s+use|allow\s+(?:the\s+word\s+)?)\s*['\u2018\u2019\"\u201c\u201d][a-zA-Z]"
    r"|\bprefer\s+(?:phrases?\s+like|words?\s+like|wording\s+like)\b"
    r")",
    re.IGNORECASE,
)

TWEAK_EDIT_RE = re.compile(
    r"(?:"
    r"\b(?:make|keep|rewrite|revise|edit|tweak|shorten)\s+(?:it|this|that|the\s+(?:draft|post|announcement|tone|style)|my\s+(?:posts?|drafts?|style))\b.*\b(?:shorter|longer|concise|brief|punchier|punchy|more\s+human|less\s+corporate|less\s+stiff|less\s+formal|more\s+casual|more\s+professional|warmer|upbeat|funnier|funny|wittier|humorous|dramatic|bullet\s*points?|bullets|paragraphs?|broetry|emojis?|hashtags?|tags?)\b"
    r"|\bsound\s+(?:more\s+human|less\s+corporate|less\s+stiff|more\s+natural|more\s+conversational|less\s+robotic|warmer|more\s+upbeat|funnier|wittier)\b"
    r"|\b(?:format\s+(?:it\s+)?(?:as|with)|switch\s+(?:back\s+)?to)\s+(?:(?:a|the|single|concise|short|upbeat)\s+)*(?:bullet\s*points?|bullets|paragraphs?(?:\s+format)?|broetry)\b"
    r"|\b(?:allow|enable|disable|strip|remove|add|include|append|insert|put)\s+(?:all\s+|some\s+|a\s+few\s+|more\s+|relevant\s+|related\s+|topical\s+|fitting\s+)?(?:emojis?|hashtags?|tags?)\b"
    r")",
    re.IGNORECASE,
)

TWEAK_DRAFT_SHORT_RE = re.compile(
    r"^\s*(?:please\s+)?(?:shorter|much\s+shorter|more\s+concise|make\s+it\s+\w+|tone\s+it\s+down|too\s+corporate\b.*|try\s+bullet\s*points?\b.*|use\s+bullets?\b.*|remove\s+(?:the\s+)?hashtags?\b.*|no\s+emojis?\b.*)\s*[.!]*$",
    re.IGNORECASE,
)

TWEAK_FOLLOWUP_EDIT_RE = re.compile(
    r"(?:"
    r"^\s*(?:please\s+)?(?:let'?s\s+|can\s+you\s+|could\s+you\s+|would\s+you\s+|now\s+)?(?:also\s+)?(?:add|include|append|insert|put|remove|drop|delete|cut|strip|change|replace|swap|adjust|tweak|modify|update|rewrite|rephrase|revise|shorten|expand|soften|polish|spice\s+up|combine|merge|weave|blend|make\s+(?:it|this|that))\b"
    r"|\b(?:hashtags?|tags?)\s+(?:of|for|about|on)\s+(?:related|relevant|similar|these|those|the)\b"
    r"|\badd\s+(?:some\s+|a\s+few\s+|more\s+|relevant\s+|related\s+|extra\s+|additional\s+|linkedin\s+)?(?:hashtags?|tags?|emojis?|humor|wit|a\s+hook|a\s+cta)\b"
    r"|\b(?:combine|merge|weave|blend|tie|stitch)\s+(?:them|these|those|both|all(?:\s+of\s+(?:them|those))?|my\s+(?:thoughts|sentences|points|ideas))\b"
    r")",
    re.IGNORECASE,
)

TROPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "career_transition_euphemism",
        re.compile(
            r"(?:\bstepping\s+back\b|\bfractional\s+advisory\b|\bexplore\s+(?:new\s+|fractional\s+)?(?:advisory\s+)?(?:opportunities|horizons)\b|\bafter\s+(?:an\s+incredible\s+chapter|\d+\s+impactful\s+years)\b|\bwrapping\s+up\s+my\s+chapter\b|\bonto\s+the\s+next\s+chapter\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "workload_red_flag",
        re.compile(
            r"(?:\bfast[- ]paced\b|\bmissionary\s+family\b|\blike\s+a\s+family\b|\brockstar\b|\b10x\s+ninja\b|\bwork\s+hard\s+play\s+hard\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "scope_creep",
        re.compile(
            r"(?:\bwears?\s+many\s+hats\b|\bthrives?\s+in\s+ambiguity\b|\brockstar\s+generalist\b|\bno\s+two\s+days\s+are\s+the\s+same\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "humblebrag_opener",
        re.compile(
            r"(?:\bhumbled\s+and\s+honored\b|\bhonored\s+and\s+humbled\b|\bhumbled\s+to\s+(?:share|announce|be|receive)\b|\bbeyond\s+(?:humbled|honored|grateful|blessed)\b|\bdeeply\s+(?:humbled|honored)\b|\binvited\s+to\s+speak\s+on\s+a\s+panel\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "mass_recruiter_template",
        re.compile(
            r"(?:\byour\s+profile\s+(?:really\s+)?stood\s+out\b|\bcame\s+across\s+your\s+(?:impressive\s+)?profile\b|\bcontract[- ]to[- ]hire\b.*\bmatches\s+your\s+skills\b|\bimpressed\s+by\s+your\s+background\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "reorg_layoff_speak",
        re.compile(
            r"(?:\bstreamlining\s+overlapping\s+functions\b|\bsunsetting\s+(?:legacy|redundant)\s+charters\b|\bholistic\s+portfolio\s+review\b|\bposition\s+our\s+organization\s+for\s+long[- ]term\b|\bright[- ]sizing\s+(?:our\s+|cross[- ]functional\s+)?(?:workforce|organization|teams?|pods?)\b|\bdifficult\s+decision\s+to\s+(?:reduce|part\s+ways|eliminate)\b|\bimpacted\s+colleagues\b)",
            re.IGNORECASE,
        ),
    ),
    (
        "satirical_broetry_bait",
        re.compile(
            r"(?:\beveryone\s+in\s+the\s+(?:coffee\s+shop|room|plane|restaurant)\s+clapped\b|\bmy\s+\d+[- ]year[- ]old\s+asked\s+me\b|\bagree\?\s*(?:\U0001F447|thoughts\?)?|\bthoughts\?\s*\U0001F447|\bread\s+that\s+again\b|\blet\s+that\s+sink\s+in\b)",
            re.IGNORECASE,
        ),
    ),
)

LINKEDIN_BUZZ_RE = re.compile(
    r"(?:\bthrilled\s+to\s+(?:announce|share)\b"
    r"|\bexcited\s+to\s+(?:announce|share)\b"
    r"|\bdelighted\s+to\s+(?:announce|share)\b"
    r"|\bproud\s+to\s+(?:announce|share)\b"
    r"|\bsynergistic\s+transformation\b"
    r"|\bthought\s+leadership\b"
    r"|\bparadigm\s+shift\b"
    r"|\bgrateful\s+for\s+the\s+journey\b"
    r"|(?:^|(?<=[\s([\{\"'\u201c\u2018]))#(?:blessed|grateful|leadership|growthmindset|innovation|opentowork)\b)",
    re.IGNORECASE,
)


def _is_spaced_broetry(text: str) -> bool:
    blocks = [blk.strip() for blk in re.split(r"\n\s*\n", text.strip()) if blk.strip()]
    if len(blocks) < 4:
        return False
    short_blocks = sum(1 for blk in blocks if len(blk) < 90 and "\n" not in blk)
    return short_blocks >= 3


def detect_human_or_linkedin_style(
    message: str,
    has_previous_draft: bool = False,
) -> dict[str, Any]:
    """Classify a conversational user turn into style, tweak, or publish intent.

    Call this tool at the start of every conversational turn to determine
    whether the user provided candid daily speech to polish into LinkedIn
    language (``HUMAN_DAILY``), corporate LinkedIn speak to decode into plain
    English (``LINKEDIN_SPEAK``), a wording preference adjustment to persist
    (``STYLE_TWEAK``), or a request to publish to LinkedIn
    (``PUBLISH_REQUEST``).

    Args:
        message: Raw conversational text sent by the user (2 to 6,000 chars).
        has_previous_draft: Whether an active converted draft already exists in
            the current session for short follow-up edit commands.

    Returns:
        Dictionary matching ``DetectStyleResponse`` with ``status``,
        ``detected_style``, ``confidence``, ``detected_signals``, and
        ``recovery_instructions``.
    """
    try:
        if not isinstance(message, str) or not isinstance(has_previous_draft, bool):
            raise ValueError(
                "message must be a string and has_previous_draft must be a boolean."
            )
        cleaned = message.strip()
        req = DetectStyleRequest(
            message=cleaned,
            has_previous_draft=has_previous_draft,
        )
        text = req.message
        text_without_urls = URL_SPAN_RE.sub(" ", text)

        if (
            PUBLISH_POS_RE.search(text)
            and not PUBLISH_NEG_RE.search(text)
            and not text.rstrip().endswith("?")
        ):
            return DetectStyleResponse(
                status="ok",
                detected_style=DetectedStyle.PUBLISH_REQUEST,
                confidence=0.96,
                detected_signals=["publish_intent"],
                recovery_instructions=[],
            ).model_dump()

        tweak_signals: list[str] = []
        if TWEAK_PERSIST_RE.search(text):
            tweak_signals.append("preference_persistence_directive")
        if TWEAK_NEG_VOCAB_RE.search(text):
            tweak_signals.append("negative_vocabulary_constraint")
        if TWEAK_EDIT_RE.search(text):
            tweak_signals.append("style_revision_directive")
        if req.has_previous_draft and (
            TWEAK_DRAFT_SHORT_RE.search(text)
            or (len(text) <= 240 and TWEAK_FOLLOWUP_EDIT_RE.search(text))
        ):
            tweak_signals.append("short_draft_adjustment")

        if tweak_signals:
            confidence = round(min(0.98, 0.88 + 0.04 * len(tweak_signals)), 2)
            return DetectStyleResponse(
                status="ok",
                detected_style=DetectedStyle.STYLE_TWEAK,
                confidence=confidence,
                detected_signals=tweak_signals,
                recovery_instructions=[],
            ).model_dump()

        linkedin_signals: list[str] = []
        for trope_name, trope_re in TROPE_PATTERNS:
            if trope_re.search(text_without_urls):
                linkedin_signals.append(trope_name)

        if LINKEDIN_BUZZ_RE.search(text_without_urls) and not linkedin_signals:
            linkedin_signals.append("corporate_buzzword_opener")

        if _is_spaced_broetry(text) and "satirical_broetry_bait" not in linkedin_signals:
            linkedin_signals.append("satirical_broetry_bait")

        if linkedin_signals:
            confidence = round(min(0.99, 0.88 + 0.05 * len(linkedin_signals)), 2)
            return DetectStyleResponse(
                status="ok",
                detected_style=DetectedStyle.LINKEDIN_SPEAK,
                confidence=confidence,
                detected_signals=linkedin_signals,
                recovery_instructions=[],
            ).model_dump()

        return DetectStyleResponse(
            status="ok",
            detected_style=DetectedStyle.HUMAN_DAILY,
            confidence=0.90,
            detected_signals=["candid_human_expression"],
            recovery_instructions=[],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="detect_human_or_linkedin_style",
            **error_payload,
        )
        return DetectStyleResponse(
            status="recoverable_error",
            detected_style=DetectedStyle.HUMAN_DAILY,
            confidence=0.0,
            detected_signals=["invalid_message_length"],
            recovery_instructions=[
                f"Input validation failed ({scrubbed_err}). Please provide a "
                "message between 2 and 6000 characters and try again."
            ],
        ).model_dump()
