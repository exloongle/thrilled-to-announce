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
"""Root ADK App configuring compaction, context cache, resumability & plugins."""

from __future__ import annotations

from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps.app import App, EventsCompactionConfig, ResumabilityConfig

from thrilled_to_announce.agent import build_agent_hierarchy
from thrilled_to_announce.config.secrets import get_secret_resolver
from thrilled_to_announce.config.settings import get_settings
from thrilled_to_announce.observability.logging import configure_logging
from thrilled_to_announce.observability.tracing import configure_tracing
from thrilled_to_announce.plugins import create_default_plugins


def create_adk_app() -> App:
    """Create the production ADK ``App`` with full observability and guardrail plugins."""
    settings = get_settings()
    configure_logging(
        project_id=settings.google_cloud_project,
    )
    configure_tracing(
        enable_cloud_export=settings.otel_export_to_cloud,
        project_id=settings.google_cloud_project,
    )
    get_secret_resolver().resolve_auth_mode(settings=settings)

    return App(
        name=settings.app_name,
        root_agent=build_agent_hierarchy(),
        plugins=create_default_plugins(settings=settings),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=4,
            overlap_size=1,
            token_threshold=6000,
            event_retention_size=4,
        ),
        context_cache_config=ContextCacheConfig(
            cache_intervals=5,
            ttl_seconds=1800,
            min_tokens=1024,
        ),
        resumability_config=ResumabilityConfig(
            is_resumable=True,
        ),
    )


app = create_adk_app()
