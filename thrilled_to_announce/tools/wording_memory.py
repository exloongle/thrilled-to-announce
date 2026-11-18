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
"""Persistent user wording preference retrieval and async consolidation tools."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from thrilled_to_announce.memory.style_store import (
    get_default_style_store,
    get_structured_logger,
    normalize_preferred_format,
    scrub_pii_text,
)
from thrilled_to_announce.schemas.contracts import (
    PreferenceLookupRequest,
    PreferenceLookupResponse,
    SavePreferenceRequest,
    SavePreferenceResponse,
    WordingPreferenceProfile,
)


def retrieve_wording_preferences(
    user_id: str = "default_user",
    query_text: str = "",
    top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve saved user wording preferences and top-k similar idiom examples.

    Call this tool before drafting a ``Human -> LinkedIn`` announcement so the
    generated draft honors persisted ``banned_words``, ``avoid_emojis``,
    ``keep_concise``, ``preferred_format``, ``tone_summary``, and semantically
    similar ``IdiomExample`` conversions across sessions.

    Args:
        user_id: Unique identifier of the user whose style profile is loaded.
        query_text: Current user draft or topic used for cosine vector lookup.
        top_k: Maximum number of ranked ``IdiomExample`` items to retrieve
            (between 1 and 10).

    Returns:
        Dictionary matching ``PreferenceLookupResponse`` containing ``status``,
        ``profile``, ``similar_examples``, and ``recovery_instructions``.
    """
    try:
        if (
            not isinstance(user_id, str)
            or not isinstance(query_text, str)
            or isinstance(top_k, bool)
            or not isinstance(top_k, int)
        ):
            raise ValueError(
                "user_id and query_text must be strings and top_k must be an integer."
            )
        scrubbed_query, _, _ = scrub_pii_text(query_text)
        req = PreferenceLookupRequest(
            user_id=user_id.strip(),
            query_text=scrubbed_query,
            top_k=top_k,
        )
        store = get_default_style_store()
        profile = store.get_profile(user_id=req.user_id)
        examples = store.search_similar_examples(
            user_id=req.user_id,
            query_text=req.query_text,
            top_k=req.top_k,
        )
        return PreferenceLookupResponse(
            status="ok",
            profile=profile,
            similar_examples=examples,
            recovery_instructions=[],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="retrieve_wording_preferences",
            **error_payload,
        )
        safe_uid = (
            user_id.strip()[:128]
            if isinstance(user_id, str) and user_id.strip()
            else "default_user"
        )
        return PreferenceLookupResponse(
            status="recoverable_error",
            profile=WordingPreferenceProfile(user_id=safe_uid),
            similar_examples=[],
            recovery_instructions=[
                f"Preference lookup failed ({scrubbed_err}). Please check "
                "user_id (1-128 chars) and top_k (1-10) and try again."
            ],
        ).model_dump()


def save_wording_preference(
    user_id: str = "default_user",
    raw_instruction: str = "",
    preferred_phrases: list[str] | None = None,
    banned_words: list[str] | None = None,
    preferred_format: str | None = None,
    avoid_emojis: bool | None = None,
    keep_concise: bool | None = None,
    preference_instruction: str | None = None,
    approved_human_example: str | None = None,
    approved_linkedin_example: str | None = None,
) -> dict[str, Any]:
    """Parse, scrub, consolidate, and asynchronously persist user wording rules.

    Call this tool whenever the user expresses a wording tweak, banned buzzword,
    emoji rule, length preference, or default formatting change (such as
    ``STYLE_TWEAK`` turns) so future sessions automatically apply the updated
    ``WordingPreferenceProfile`` without blocking the conversational reply.

    Args:
        user_id: Unique identifier of the user whose style profile is updated.
        raw_instruction: Natural-language wording preference directive from the
            user (3 to 2,000 characters).
        preferred_phrases: Optional list of favored phrases to record in vector
            style memory and ``tone_summary``.
        banned_words: Optional explicit list of forbidden words to merge and
            deduplicate in lowercase.
        preferred_format: Optional layout override (``'paragraph'``,
            ``'bullets'``, ``'broetry'``, or ``'short_punchy'``).
        avoid_emojis: Optional boolean flag controlling whether all emojis are
            stripped from LinkedIn drafts.
        keep_concise: Optional boolean flag requesting concise 1-2 sentence
            LinkedIn conversions.
        preference_instruction: Optional alias for ``raw_instruction`` matching
            ``SavePreferenceRequest.preference_instruction`` directly.
        approved_human_example: Optional plain-human source sentence paired with
            ``approved_linkedin_example`` for dense vector indexing.
        approved_linkedin_example: Optional user-approved LinkedIn phrasing
            indexed in ``VectorStyleMemoryStore`` for cosine retrieval.

    Returns:
        Dictionary matching ``SavePreferenceResponse`` with ``status``,
        ``persistence_mode``, ``updated_profile``, and ``recovery_instructions``.
    """
    try:
        if not isinstance(user_id, str):
            raise ValueError("user_id must be a string.")
        if banned_words is not None and (
            isinstance(banned_words, (str, bytes, bytearray))
            or not isinstance(banned_words, Sequence)
            or any(not isinstance(item, str) for item in banned_words)
        ):
            raise ValueError("banned_words must be a list of strings.")
        if preferred_phrases is not None and (
            isinstance(preferred_phrases, (str, bytes, bytearray))
            or not isinstance(preferred_phrases, Sequence)
            or any(not isinstance(item, str) for item in preferred_phrases)
        ):
            raise ValueError("preferred_phrases must be a list of strings.")
        if avoid_emojis is not None and not isinstance(avoid_emojis, bool):
            raise ValueError("avoid_emojis must be a boolean or None.")
        if keep_concise is not None and not isinstance(keep_concise, bool):
            raise ValueError("keep_concise must be a boolean or None.")

        effective_raw = (
            preference_instruction
            if preference_instruction is not None
            else raw_instruction
        )
        if not isinstance(effective_raw, str):
            raise ValueError("raw_instruction must be a string.")
        scrubbed_instr, _, _ = scrub_pii_text(effective_raw.strip())
        normalized_fmt, concise_from_fmt = normalize_preferred_format(preferred_format)
        effective_concise = (
            keep_concise
            if keep_concise is not None
            else (True if concise_from_fmt else None)
        )
        scrubbed_human = (
            scrub_pii_text(approved_human_example)[0]
            if approved_human_example is not None
            else None
        )
        scrubbed_linkedin = (
            scrub_pii_text(approved_linkedin_example)[0]
            if approved_linkedin_example is not None
            else None
        )
        cleaned_banned = [
            scrub_pii_text(item.strip().lower())[0]
            for item in (banned_words or [])
            if item.strip()
        ]
        cleaned_phrases = (
            [item.strip() for item in preferred_phrases if item.strip()]
            if preferred_phrases is not None
            else None
        )

        req = SavePreferenceRequest(
            user_id=user_id.strip(),
            preference_instruction=scrubbed_instr,
            banned_words=cleaned_banned,
            avoid_emojis=avoid_emojis,
            keep_concise=effective_concise,
            preferred_format=normalized_fmt,
            approved_human_example=scrubbed_human,
            approved_linkedin_example=scrubbed_linkedin,
        )

        store = get_default_style_store()
        updated_profile, persistence_mode = store.schedule_async_consolidation(
            user_id=req.user_id,
            raw_instruction=req.preference_instruction,
            banned_words=req.banned_words,
            preferred_phrases=cleaned_phrases,
            avoid_emojis=req.avoid_emojis,
            keep_concise=req.keep_concise,
            preferred_format=req.preferred_format,
            approved_human_example=req.approved_human_example,
            approved_linkedin_example=req.approved_linkedin_example,
        )
        return SavePreferenceResponse(
            status="ok",
            persistence_mode=persistence_mode,
            updated_profile=updated_profile,
            recovery_instructions=[],
        ).model_dump()
    except Exception as exc:
        scrubbed_err, _, _ = scrub_pii_text(str(exc))
        error_payload = {"status": "error", "error_message": scrubbed_err}
        get_structured_logger(__name__).warning(
            "tool_recoverable_error",
            tool="save_wording_preference",
            **error_payload,
        )
        safe_uid = (
            user_id.strip()[:128]
            if isinstance(user_id, str) and user_id.strip()
            else "default_user"
        )
        fallback_profile = WordingPreferenceProfile(user_id=safe_uid)
        try:
            fallback_profile = get_default_style_store().get_profile(user_id=safe_uid)
        except Exception:
            pass
        return SavePreferenceResponse(
            status="recoverable_error",
            persistence_mode="none",
            updated_profile=fallback_profile,
            recovery_instructions=[
                f"Could not save preference ({scrubbed_err}). Please provide a "
                "clear wording instruction (3-2000 characters) and valid "
                "preferred_format ('paragraph', 'bullets', or 'broetry') and try again."
            ],
        ).model_dump()
