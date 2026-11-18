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
"""ADK policy and telemetry plugins package."""

from __future__ import annotations

from google.adk.plugins.base_plugin import BasePlugin

from thrilled_to_announce.config.settings import AppSettings
from thrilled_to_announce.observability.pii_scrubber import PiiScrubber
from thrilled_to_announce.plugins.policy_plugin import (
    COACH_REDIRECT_SUGGESTED_REPLIES,
    DOXXING_REFUSAL_TEXT,
    NDA_REFUSAL_TEXT,
    PROMPT_INJECTION_REFUSAL_TEXT,
    BridgeBurnAndNdaPolicyPlugin,
    PolicyDecision,
    create_model_armor_plugin,
    default_safety_settings,
    evaluate_input_policy,
    evaluate_output_policy,
    evaluate_topic_scope,
    off_topic_coach_redirect,
    screen_model_response_with_model_armor,
    screen_user_prompt_with_model_armor,
)
from thrilled_to_announce.plugins.telemetry_plugin import (
    IntentOutcomeContext,
    IntentOutcomeTelemetryPlugin,
    record_intent_and_outcome,
)


def create_default_plugins(
    *,
    scrubber: PiiScrubber | None = None,
    settings: AppSettings | None = None,
) -> list[BasePlugin]:
    """Create the ordered default ADK plugin stack (Telemetry, Policy, optional Model Armor)."""
    plugins: list[BasePlugin] = [
        IntentOutcomeTelemetryPlugin(),
        BridgeBurnAndNdaPolicyPlugin(scrubber=scrubber),
    ]
    model_armor_plugin = create_model_armor_plugin(settings=settings)
    if model_armor_plugin is not None:
        plugins.append(model_armor_plugin)
    return plugins


__all__ = [
    "BridgeBurnAndNdaPolicyPlugin",
    "COACH_REDIRECT_SUGGESTED_REPLIES",
    "DOXXING_REFUSAL_TEXT",
    "IntentOutcomeContext",
    "IntentOutcomeTelemetryPlugin",
    "NDA_REFUSAL_TEXT",
    "PROMPT_INJECTION_REFUSAL_TEXT",
    "PolicyDecision",
    "create_default_plugins",
    "create_model_armor_plugin",
    "default_safety_settings",
    "evaluate_input_policy",
    "evaluate_output_policy",
    "evaluate_topic_scope",
    "off_topic_coach_redirect",
    "record_intent_and_outcome",
    "screen_model_response_with_model_armor",
    "screen_user_prompt_with_model_armor",
]
