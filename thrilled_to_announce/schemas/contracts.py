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
"""Shared Pydantic v2 contracts across tools, memory, server, UI, and evals."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DetectedStyle(str, Enum):
    """Auto-detected intent and language style of a conversational user message."""

    HUMAN_DAILY = "HUMAN_DAILY"
    LINKEDIN_SPEAK = "LINKEDIN_SPEAK"
    STYLE_TWEAK = "STYLE_TWEAK"
    PUBLISH_REQUEST = "PUBLISH_REQUEST"


class PreferredFormat(str, Enum):
    """Formatting layout preference stored in persistent user style memory."""

    PARAGRAPH = "paragraph"
    BULLETS = "bullets"
    BROETRY = "broetry"


class WordingPreferenceProfile(BaseModel):
    """Persistent user wording preference profile loaded on every conversion."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    tone_summary: str = Field(
        default="balanced professional polish",
        description="Natural-language summary of desired LinkedIn voice.",
    )
    banned_words: list[str] = Field(
        default_factory=list,
        description="Lowercase words the user asked never to include (e.g. 'exhausted', 'humbled').",
    )
    avoid_emojis: bool = Field(
        default=False,
        description="Whether to strip all emojis from converted LinkedIn drafts.",
    )
    keep_concise: bool = Field(
        default=False,
        description="Whether the user requested shorter, punchier 1-2 sentence conversions.",
    )
    preferred_format: PreferredFormat = Field(default=PreferredFormat.PARAGRAPH)
    has_custom_preferences: bool = Field(default=False)


class IdiomExample(BaseModel):
    """Vector-indexed bidirectional idiom example stored in SQLite memory."""

    model_config = ConfigDict(extra="forbid")

    entry_id: str
    user_id: str
    human_text: str
    linkedin_text: str
    similarity_score: float = 0.0


class DetectStyleRequest(BaseModel):
    """Validated input schema for ``detect_human_or_linkedin_style``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=2, max_length=6000)
    has_previous_draft: bool = Field(default=False)


class DetectStyleResponse(BaseModel):
    """Validated output schema for ``detect_human_or_linkedin_style``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "recoverable_error"] = "ok"
    detected_style: DetectedStyle = DetectedStyle.HUMAN_DAILY
    confidence: float = Field(ge=0.0, le=1.0, default=0.85)
    detected_signals: list[str] = Field(default_factory=list)
    recovery_instructions: list[str] = Field(default_factory=list)


class PreferenceLookupRequest(BaseModel):
    """Validated input schema for ``retrieve_wording_preferences``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    query_text: str = Field(default="", max_length=4000)
    top_k: int = Field(default=3, ge=1, le=10)


class PreferenceLookupResponse(BaseModel):
    """Validated output schema for ``retrieve_wording_preferences``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "recoverable_error"] = "ok"
    profile: WordingPreferenceProfile
    similar_examples: list[IdiomExample] = Field(default_factory=list)
    recovery_instructions: list[str] = Field(default_factory=list)


class SavePreferenceRequest(BaseModel):
    """Validated input schema for ``save_wording_preference``."""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    preference_instruction: str = Field(..., min_length=3, max_length=2000)
    banned_words: list[str] = Field(default_factory=list)
    avoid_emojis: Optional[bool] = Field(default=None)
    keep_concise: Optional[bool] = Field(default=None)
    preferred_format: Optional[PreferredFormat] = Field(default=None)
    approved_human_example: Optional[str] = Field(default=None)
    approved_linkedin_example: Optional[str] = Field(default=None)


class SavePreferenceResponse(BaseModel):
    """Validated output schema for ``save_wording_preference``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "recoverable_error"] = "ok"
    persistence_mode: Literal[
        "async_background_queued", "synchronous_fallback", "none"
    ] = "async_background_queued"
    updated_profile: WordingPreferenceProfile
    recovery_instructions: list[str] = Field(default_factory=list)


class ToneValidationRequest(BaseModel):
    """Validated input schema for ``score_tone_and_bridge_burn_risk``."""

    model_config = ConfigDict(extra="forbid")

    candidate_text: str
    target_style: Literal["LINKEDIN_SPEAK", "HUMAN_DAILY"] = "LINKEDIN_SPEAK"
    banned_words: list[str] = Field(default_factory=list)
    avoid_emojis: bool = Field(default=False)
    max_characters: int = Field(default=3000, ge=50, le=5000)


class ToneValidationResponse(BaseModel):
    """Validated output schema for ``score_tone_and_bridge_burn_risk``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "recoverable_error"]
    character_count: int
    max_characters: int
    bridge_burn_risk_score: int = Field(ge=0, le=100)
    flagged_terms: list[str] = Field(default_factory=list)
    recovery_instructions: list[str] = Field(default_factory=list)


class PublishFeedRequest(BaseModel):
    """Validated input schema for HITL ``publish_to_linkedin_feed``."""

    model_config = ConfigDict(extra="forbid")

    post_content: str = Field(..., min_length=5, max_length=3000)
    user_id: str = Field(default="default_user", min_length=1, max_length=128)


class PublishFeedResponse(BaseModel):
    """Validated output schema for HITL ``publish_to_linkedin_feed``."""

    model_config = ConfigDict(extra="forbid")

    status: Literal[
        "published",
        "awaiting_human_confirmation",
        "cancelled_by_human",
        "recoverable_error",
    ]
    publish_id: str
    preview_text: str
    message: str
    recovery_instructions: list[str] = Field(default_factory=list)


class WelcomeResponse(BaseModel):
    """HTTP response schema for ``GET /api/welcome``."""

    model_config = ConfigDict(extra="forbid")

    greeting: str
    starter_prompts: list[str]


class ChatTurnRequest(BaseModel):
    """HTTP request schema for ``POST /api/chat``."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(..., min_length=2, max_length=6000)
    user_id: str = Field(default="default_user", min_length=1, max_length=128)
    session_id: Optional[str] = Field(default=None)


class ChatTurnResponse(BaseModel):
    """HTTP response schema for ``POST /api/chat``."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    detected_style: DetectedStyle
    label: str
    reply_text: str
    coach_intro: Optional[str] = None
    proactive_prompt: Optional[str] = None
    suggested_replies: list[str] = Field(default_factory=list)
    preference_saved: bool = False
    used_saved_preference: bool = False
    can_post_to_linkedin: bool = False
    hitl_required: bool = False
    pending_publish_id: Optional[str] = None
    guardrail_blocked: bool = False
    pii_redactions_count: int = 0
    bridge_burn_risk_score: int = 0


class PublishConfirmHttpRequest(BaseModel):
    """HTTP request schema for ``POST /api/publish/confirm``."""

    model_config = ConfigDict(extra="forbid")

    publish_id: str = Field(..., min_length=3, max_length=128)
    confirmed: bool
