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
"""Tone constraint, length, hashtag, and career bridge-burn risk validator tool."""

from __future__ import annotations

from collections.abc import Sequence
import re
from typing import Any, Literal

from thrilled_to_announce.memory.style_store import (
    get_structured_logger,
    scrub_pii_text,
)
from thrilled_to_announce.schemas.contracts import (
    ToneValidationRequest,
    ToneValidationResponse,
)

URL_SPAN_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
HASHTAG_RE = re.compile(r"""(?:^|(?<=[\s([{"'\u201c\u2018]))#[A-Za-z][A-Za-z0-9_]*""")

EMOJI_RE = re.compile(
    r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U00002B50\U00002B55"
    r"\U0000231A-\U0000231B\U000023E9-\U000023F3\U000023F8-\U000023FA"
    r"\U0001F1E6-\U0001F1FF]"
)

_SEVERE_RISK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("idiot", re.compile(r"\bidiot(?:s|ic)?\b", re.IGNORECASE)),
    ("incompetent", re.compile(r"\bincompetent\b", re.IGNORECASE)),
    ("trash", re.compile(r"\btrash(?:y)?\b", re.IGNORECASE)),
    ("garbage", re.compile(r"\bgarbage\b", re.IGNORECASE)),
    ("moron", re.compile(r"\bmoron(?:s|ic)?\b", re.IGNORECASE)),
    ("pathetic", re.compile(r"\bpathetic\b", re.IGNORECASE)),
    ("dumpster_fire", re.compile(r"\bdumpster\s+fire\b", re.IGNORECASE)),
    ("clown", re.compile(r"\bclowns?\b", re.IGNORECASE)),
    ("hostile_hate", re.compile(r"\bhate\s+(?:my|this|our)\b", re.IGNORECASE)),
    ("ruined", re.compile(r"\bruined\b", re.IGNORECASE)),
    ("sucks", re.compile(r"\bsucks\b", re.IGNORECASE)),
    ("crap", re.compile(r"\bcrap\b", re.IGNORECASE)),
    ("damn", re.compile(r"\bdamn\b", re.IGNORECASE)),
    ("wtf", re.compile(r"\bwtf\b", re.IGNORECASE)),
    ("doxxing", re.compile(r"\bdoxx\w*\b", re.IGNORECASE)),
    (
        "personal_address",
        re.compile(r"\b(?:personal|home)\s+address\b", re.IGNORECASE),
    ),
    (
        "non_public_disclosure",
        re.compile(
            r"(?:\bunreleased\s+q[1-4]\b|\bproject\s+codename\b|\bconfiden[t]ial\b)",
            re.IGNORECASE,
        ),
    ),
    ("unredacted_ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)

_CANDID_LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("boring", re.compile(r"\bboring\b", re.IGNORECASE)),
    ("tired_of", re.compile(r"\btired\s+of\b", re.IGNORECASE)),
    ("micromanage", re.compile(r"\bmicromanag\w*\b", re.IGNORECASE)),
    ("my_boss", re.compile(r"\bmy\s+boss\b", re.IGNORECASE)),
    ("desperate", re.compile(r"\bdesperate\b", re.IGNORECASE)),
    (
        "ran_out_of_funds",
        re.compile(r"\bran\s+out\s+of\s+(?:money|budget)\b", re.IGNORECASE),
    ),
    ("stop_putting_me", re.compile(r"\bstop\s+putting\s+me\b", re.IGNORECASE)),
    ("way_more_money", re.compile(r"\bway\s+more\s+money\b", re.IGNORECASE)),
    ("cash", re.compile(r"\bcash\b", re.IGNORECASE)),
    ("broke_prod", re.compile(r"\bbroke\s+prod\b", re.IGNORECASE)),
    ("no_idea", re.compile(r"\bno\s+idea\b", re.IGNORECASE)),
    ("glued", re.compile(r"\bglued\b", re.IGNORECASE)),
    ("burned_out", re.compile(r"\bburned\s+out\b", re.IGNORECASE)),
)

_MILD_CONTEXT_RE = re.compile(
    r"\b(?:incident|outage|postmortem|restructuring)\b",
    re.IGNORECASE,
)


def _normalize_target_style(raw_style: str) -> Literal["LINKEDIN_SPEAK", "HUMAN_DAILY"]:
    cleaned = (raw_style or "").strip().upper().replace("-", "_").replace(" ", "_")
    if cleaned in {"LINKEDIN_SPEAK", "LINKEDIN", "LINKEDIN_STYLE"}:
        return "LINKEDIN_SPEAK"
    if cleaned in {"HUMAN_DAILY", "HUMAN", "PLAIN_ENGLISH"}:
        return "HUMAN_DAILY"
    raise ValueError(
        f"Invalid target_style '{raw_style}'. Valid options are "
        "'LINKEDIN_SPEAK' or 'HUMAN_DAILY'."
    )


def score_tone_and_bridge_burn_risk(
    candidate_text: str,
    target_style: str = "HUMAN_DAILY",
    banned_words: list[str] | None = None,
    avoid_emojis: bool = False,
    max_characters: int = 3000,
) -> dict[str, Any]:
    """Validate draft length, hashtag count, style rules, and bridge-burn risk.

    Call this tool in the self-evaluation critic pass before returning or
    publishing a converted draft. When constraints fail (for example ``> 3000``
    characters, ``> 5`` hashtags, user-banned words, emojis when
    ``avoid_emojis=True``, or ``bridge_burn_risk_score > 15``), it returns
    ``status="recoverable_error"`` with concrete ``recovery_instructions`` so
    the agent can revise and re-validate without raising an exception.

    Args:
        candidate_text: Draft conversion text to inspect for safety and style
            fidelity.
        target_style: Target output register (``'LINKEDIN_SPEAK'`` for polished
            LinkedIn announcements or ``'HUMAN_DAILY'`` for plain-English
            subtext decodings).
        banned_words: Optional list of user-forbidden words that must not appear
            in ``candidate_text``.
        avoid_emojis: Whether to flag any Unicode emoji characters present in
            ``candidate_text``.
        max_characters: Maximum allowed character count (default ``3000``,
            valid range ``50`` to ``5000``).

    Returns:
        Dictionary matching ``ToneValidationResponse`` containing ``status``,
        ``character_count``, ``max_characters``, ``bridge_burn_risk_score``,
        ``flagged_terms``, and ``recovery_instructions``.
    """
    try:
        if not isinstance(candidate_text, str):
            raise ValueError("candidate_text must be a string.")
        if not isinstance(target_style, str):
            raise ValueError("target_style must be a string.")
        if banned_words is not None and (
            isinstance(banned_words, (str, bytes, bytearray))
            or not isinstance(banned_words, Sequence)
            or any(not isinstance(item, str) for item in banned_words)
        ):
            raise ValueError("banned_words must be a list of strings.")
        if not isinstance(avoid_emojis, bool):
            raise ValueError("avoid_emojis must be a boolean.")
        if isinstance(max_characters, bool) or not isinstance(max_characters, int):
            raise ValueError("max_characters must be an integer between 50 and 5000.")

        normalized_style = _normalize_target_style(target_style)
        req = ToneValidationRequest(
            candidate_text=candidate_text,
            target_style=normalized_style,
            banned_words=[
                w.strip().lower() for w in (banned_words or []) if w.strip()
            ],
            avoid_emojis=avoid_emojis,
            max_characters=max_characters,
        )

        text = req.candidate_text
        char_count = len(text)
        if not text.strip():
            return ToneValidationResponse(
                status="recoverable_error",
                character_count=char_count,
                max_characters=req.max_characters,
                bridge_burn_risk_score=0,
                flagged_terms=["empty_candidate_text"],
                recovery_instructions=[
                    "Candidate draft is empty. Please provide a non-empty draft and try again."
                ],
            ).model_dump()

        flagged_terms: list[str] = []
        recovery_instructions: list[str] = []

        if char_count > req.max_characters:
            excess = char_count - req.max_characters
            flagged_terms.append("exceeds_max_characters")
            recovery_instructions.append(
                f"Draft is {char_count} characters, exceeding the "
                f"{req.max_characters}-character limit by {excess} characters. "
                f"Please trim the draft to {req.max_characters} characters or "
                "fewer and try again."
            )

        text_without_urls = URL_SPAN_RE.sub(" ", text)
        hashtags = HASHTAG_RE.findall(text_without_urls)
        if len(hashtags) > 5:
            flagged_terms.append("excessive_hashtags")
            recovery_instructions.append(
                f"Draft contains {len(hashtags)} hashtags (maximum allowed is 5). "
                f"Please trim {len(hashtags) - 5} hashtag(s) and try again."
            )

        matched_banned: list[str] = []
        for bw in req.banned_words:
            if not bw:
                continue
            pattern = (
                re.compile(rf"\b{re.escape(bw)}\b", re.IGNORECASE)
                if bw[0].isalnum() and bw[-1].isalnum()
                else re.compile(re.escape(bw), re.IGNORECASE)
            )
            if pattern.search(text):
                matched_banned.append(bw)
                if bw not in flagged_terms:
                    flagged_terms.append(bw)

        if matched_banned:
            joined_bw = ", ".join(f"'{w}'" for w in matched_banned)
            recovery_instructions.append(
                f"Draft contains user-banned wording ({joined_bw}). Please "
                "replace those words with natural alternatives and try again."
            )

        if req.avoid_emojis and EMOJI_RE.search(text):
            flagged_terms.append("emoji_detected")
            recovery_instructions.append(
                "Draft contains emojis while avoid_emojis is enabled. Please "
                "strip all emojis and try again."
            )

        risk_score = 0
        risk_terms: list[str] = []

        for label, severe_re in _SEVERE_RISK_PATTERNS:
            if severe_re.search(text):
                risk_score += 35
                risk_terms.append(label)
                if label not in flagged_terms:
                    flagged_terms.append(label)

        if req.target_style == "LINKEDIN_SPEAK":
            for label, candid_re in _CANDID_LEAK_PATTERNS:
                if candid_re.search(text):
                    risk_score += 25
                    risk_terms.append(label)
                    if label not in flagged_terms:
                        flagged_terms.append(label)
        else:
            for label, candid_re in _CANDID_LEAK_PATTERNS:
                if label in {"broke_prod", "micromanage"} and candid_re.search(text):
                    risk_score += 25
                    risk_terms.append(label)
                    if label not in flagged_terms:
                        flagged_terms.append(label)

        mild_hits = len(set(m.lower() for m in _MILD_CONTEXT_RE.findall(text)))
        risk_score = min(100, risk_score + min(15, mild_hits * 5))

        if risk_score > 15:
            joined_risk = ", ".join(risk_terms) if risk_terms else "elevated_tone_risk"
            recovery_instructions.append(
                f"Bridge-burn risk score is {risk_score}/100 due to "
                f"{joined_risk}. Please reframe constructively without insults, "
                "blame, or raw complaints and try again."
            )

        status: Literal["ok", "recoverable_error"] = (
            "recoverable_error" if recovery_instructions else "ok"
        )
        return ToneValidationResponse(
            status=status,
            character_count=char_count,
            max_characters=req.max_characters,
            bridge_burn_risk_score=risk_score,
            flagged_terms=flagged_terms,
            recovery_instructions=recovery_instructions,
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="score_tone_and_bridge_burn_risk",
            **error_payload,
        )
        safe_max = max_characters if isinstance(max_characters, int) and 50 <= max_characters <= 5000 else 3000
        safe_len = len(candidate_text) if isinstance(candidate_text, str) else 0
        return ToneValidationResponse(
            status="recoverable_error",
            character_count=safe_len,
            max_characters=safe_max,
            bridge_burn_risk_score=0,
            flagged_terms=["validation_parameter_error"],
            recovery_instructions=[
                f"Tone validation request invalid ({scrubbed_err}). Valid options "
                "are target_style in ('LINKEDIN_SPEAK', 'HUMAN_DAILY') and "
                "max_characters between 50 and 5000. Please check parameters and try again."
            ],
        ).model_dump()
