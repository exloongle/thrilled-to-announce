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
"""Multi-agent ADK orchestration graph for Thrilled to Announce."""

from __future__ import annotations

from google.adk.agents import LlmAgent, ParallelAgent, SequentialAgent
from google.genai import types

from thrilled_to_announce.memory.session_factory import after_agent_memory_callback
from thrilled_to_announce.plugins.policy_plugin import default_safety_settings
from thrilled_to_announce.prompts.constitution import (
    COORDINATOR_INSTRUCTION,
    REACTION_SIMULATOR_INSTRUCTION,
    SELF_EVAL_AUDITOR_INSTRUCTION,
    SUBTEXT_ANALYST_INSTRUCTION,
    TRANSLATOR_INSTRUCTION,
)
from thrilled_to_announce.routing.model_router import (
    StrategicModelRouter,
    TaskComplexityTier,
)
from thrilled_to_announce.tools import (
    detect_human_or_linkedin_style_tool,
    preload_memory_tool,
    publish_to_linkedin_tool,
    retrieve_wording_preferences_tool,
    save_wording_preference_tool,
    score_tone_and_bridge_burn_risk_tool,
)


def build_agent_hierarchy(
    router: StrategicModelRouter | None = None,
) -> LlmAgent:
    """Construct the ADK multi-agent hierarchy with strategic Flash/Pro routing.

    Topology:
        coordinator_agent (Flash)
          └── conversion_pipeline (SequentialAgent)
                ├── subtext_analyst_agent (Pro)
                ├── conversion_and_reaction_crew (ParallelAgent)
                │     ├── linkedin_converter_agent (Flash)
                │     └── feed_reaction_simulator_agent (Flash)
                └── self_eval_auditor_agent (Pro)
    """
    active_router = router or StrategicModelRouter()

    coordinator_model = active_router.select_model(
        agent_role="coordinator_agent",
        tier=TaskComplexityTier.FAST_ROUTING_AND_DRAFTING,
    ).model_name
    analyst_model = active_router.select_model(
        agent_role="subtext_analyst_agent",
        tier=TaskComplexityTier.DEEP_REASONING_AND_AUDIT,
    ).model_name
    converter_model = active_router.select_model(
        agent_role="linkedin_converter_agent",
        tier=TaskComplexityTier.FAST_ROUTING_AND_DRAFTING,
    ).model_name
    reaction_model = active_router.select_model(
        agent_role="feed_reaction_simulator_agent",
        tier=TaskComplexityTier.FAST_ROUTING_AND_DRAFTING,
    ).model_name
    auditor_model = active_router.select_model(
        agent_role="self_eval_auditor_agent",
        tier=TaskComplexityTier.DEEP_REASONING_AND_AUDIT,
    ).model_name

    safety_cfg = types.GenerateContentConfig(
        safety_settings=default_safety_settings(),
    )

    subtext_analyst_agent = LlmAgent(
        name="subtext_analyst_agent",
        model=analyst_model,
        description=(
            "Extracts career intent, hidden corporate subtext, and active "
            "wording-preference constraints prior to drafting."
        ),
        instruction=SUBTEXT_ANALYST_INSTRUCTION,
        tools=[retrieve_wording_preferences_tool],
        generate_content_config=safety_cfg,
        output_key="subtext_analysis",
    )

    linkedin_converter_agent = LlmAgent(
        name="linkedin_converter_agent",
        model=converter_model,
        description=(
            "Converts Human -> LinkedIn or LinkedIn -> Plain English and "
            "applies/saves conversational wording preferences."
        ),
        instruction=TRANSLATOR_INSTRUCTION,
        tools=[
            save_wording_preference_tool,
            score_tone_and_bridge_burn_risk_tool,
        ],
        generate_content_config=safety_cfg,
        output_key="candidate_conversion",
    )

    feed_reaction_simulator_agent = LlmAgent(
        name="feed_reaction_simulator_agent",
        model=reaction_model,
        description=(
            "Evaluates recruiter/hiring-manager impression concurrently with "
            "conversion drafting."
        ),
        instruction=REACTION_SIMULATOR_INSTRUCTION,
        generate_content_config=safety_cfg,
        output_key="reaction_check",
    )

    conversion_and_reaction_crew = ParallelAgent(
        name="conversion_and_reaction_crew",
        description=(
            "Runs bidirectional conversion and recruiter-signal review in parallel."
        ),
        sub_agents=[
            linkedin_converter_agent,
            feed_reaction_simulator_agent,
        ],
    )

    self_eval_auditor_agent = LlmAgent(
        name="self_eval_auditor_agent",
        model=auditor_model,
        description=(
            "Performs self-evaluation tone, banned-word, emoji, and length "
            "auditing before returning the final concise reply."
        ),
        instruction=SELF_EVAL_AUDITOR_INSTRUCTION,
        tools=[
            score_tone_and_bridge_burn_risk_tool,
            save_wording_preference_tool,
        ],
        generate_content_config=safety_cfg,
        output_key="final_conversion",
    )

    conversion_pipeline = SequentialAgent(
        name="conversion_pipeline",
        description=(
            "Sequential pipeline: Subtext Analyst -> Parallel Conversion & "
            "Reaction Crew -> Self-Eval Tone Auditor."
        ),
        sub_agents=[
            subtext_analyst_agent,
            conversion_and_reaction_crew,
            self_eval_auditor_agent,
        ],
    )

    coordinator_agent = LlmAgent(
        name="coordinator_agent",
        model=coordinator_model,
        description=(
            "Root conversational coordinator detecting language style, loading "
            "wording memory, routing conversions, and gating HITL publishing."
        ),
        instruction=COORDINATOR_INSTRUCTION,
        tools=[
            detect_human_or_linkedin_style_tool,
            retrieve_wording_preferences_tool,
            publish_to_linkedin_tool,
            preload_memory_tool,
        ],
        sub_agents=[conversion_pipeline],
        generate_content_config=safety_cfg,
        after_agent_callback=after_agent_memory_callback,
    )

    return coordinator_agent


root_agent = build_agent_hierarchy()
