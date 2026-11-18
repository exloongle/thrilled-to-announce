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
"""Strategic Gemini model router balancing deep reasoning and low latency."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from thrilled_to_announce.config.settings import (
    DEFAULT_FLASH_MODEL,
    DEFAULT_PRO_MODEL,
    get_settings,
)
from thrilled_to_announce.observability.logging import get_logger


class TaskComplexityTier(str, Enum):
    """Workload classification tiers driving Flash vs. Pro routing."""

    FAST_ROUTING_AND_DRAFTING = "fast_routing_and_drafting"
    DEEP_REASONING_AND_AUDIT = "deep_reasoning_and_audit"


@dataclass(frozen=True)
class ModelRouteDecision:
    """Resolved model assignment and rationale emitted for observability."""

    agent_role: str
    tier: TaskComplexityTier
    model_name: str
    rationale: str


class StrategicModelRouter:
    """Routes ADK sub-agents to Flash (fast tasks) or Pro (deep reasoning/audit)."""

    FLASH_MODEL = DEFAULT_FLASH_MODEL
    PRO_MODEL = DEFAULT_PRO_MODEL

    def __init__(
        self,
        pro_model: str | None = None,
        flash_model: str | None = None,
    ) -> None:
        settings = get_settings()
        self._pro_model = pro_model or settings.pro_model or self.PRO_MODEL
        self._flash_model = flash_model or settings.flash_model or self.FLASH_MODEL

    @property
    def pro_model(self) -> str:
        """Configured Gemini Pro model identifier."""
        return self._pro_model

    @property
    def flash_model(self) -> str:
        """Configured Gemini Flash model identifier."""
        return self._flash_model

    def select_model(
        self,
        *,
        agent_role: str,
        tier: TaskComplexityTier,
    ) -> ModelRouteDecision:
        """Select ``gemini-2.5-pro`` for deep analysis/audit and ``gemini-2.5-flash`` for speed."""
        if tier == TaskComplexityTier.DEEP_REASONING_AND_AUDIT:
            decision = ModelRouteDecision(
                agent_role=agent_role,
                tier=tier,
                model_name=self._pro_model,
                rationale=(
                    "High-nuance subtext extraction and self-eval career-safety "
                    "auditing require Pro reasoning."
                ),
            )
        else:
            decision = ModelRouteDecision(
                agent_role=agent_role,
                tier=tier,
                model_name=self._flash_model,
                rationale=(
                    "Low-latency conversational routing, parallel drafting, and "
                    "reaction simulation run optimally on Flash."
                ),
            )

        get_logger(__name__).info(
            "strategic_model_routed",
            agent_role=decision.agent_role,
            tier=decision.tier.value,
            model_name=decision.model_name,
        )
        return decision
