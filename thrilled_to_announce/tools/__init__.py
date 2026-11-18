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
"""Validated ADK FunctionTool registry for Thrilled to Announce."""

from __future__ import annotations

from google.adk.tools import FunctionTool, load_memory, preload_memory
from google.adk.tools.preload_memory_tool import PreloadMemoryTool, preload_memory_tool

from thrilled_to_announce.tools.linkedin_publisher import (
    confirm_publish_decision,
    publish_to_linkedin_feed,
    publish_to_linkedin_tool,
    request_publish_approval,
)
from thrilled_to_announce.tools.style_detection import (
    detect_human_or_linkedin_style,
)
from thrilled_to_announce.tools.tone_validator import (
    score_tone_and_bridge_burn_risk,
)
from thrilled_to_announce.tools.wording_memory import (
    retrieve_wording_preferences,
    save_wording_preference,
)

detect_human_or_linkedin_style_tool = FunctionTool(detect_human_or_linkedin_style)
retrieve_wording_preferences_tool = FunctionTool(retrieve_wording_preferences)
save_wording_preference_tool = FunctionTool(save_wording_preference)
score_tone_and_bridge_burn_risk_tool = FunctionTool(score_tone_and_bridge_burn_risk)

ALL_ADK_TOOLS: tuple[FunctionTool, ...] = (
    detect_human_or_linkedin_style_tool,
    retrieve_wording_preferences_tool,
    save_wording_preference_tool,
    score_tone_and_bridge_burn_risk_tool,
    publish_to_linkedin_tool,
)

__all__ = [
    "ALL_ADK_TOOLS",
    "PreloadMemoryTool",
    "confirm_publish_decision",
    "detect_human_or_linkedin_style",
    "detect_human_or_linkedin_style_tool",
    "load_memory",
    "preload_memory",
    "preload_memory_tool",
    "publish_to_linkedin_feed",
    "publish_to_linkedin_tool",
    "request_publish_approval",
    "retrieve_wording_preferences",
    "retrieve_wording_preferences_tool",
    "save_wording_preference",
    "save_wording_preference_tool",
    "score_tone_and_bridge_burn_risk",
    "score_tone_and_bridge_burn_risk_tool",
]
