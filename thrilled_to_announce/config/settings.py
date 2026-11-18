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
"""Environment-driven application configuration with zero hardcoded project IDs or keys."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

try:
    from dotenv import load_dotenv

    if "PYTEST_CURRENT_TEST" not in os.environ and Path(".env").is_file():
        load_dotenv(dotenv_path=Path(".env"), override=False)
except ImportError:  # pragma: no cover
    pass

DEFAULT_APP_NAME = "thrilled_to_announce"
DEFAULT_GCP_LOCATION = "us-central1"
DEFAULT_SECRET_ID = "gemini-api-key"
DEFAULT_DATA_DIR = Path(".thrilled_data")
DEFAULT_FLASH_MODEL = "gemini-4-hydrogen"
DEFAULT_PRO_MODEL = "gemini-4-hydrogen"
DEFAULT_PORT = 8080

_PLACEHOLDER_PROJECT_VALUES = frozenset({
    "",
    "your_project_id",
    "<your_project_id>",
    "your-project-id",
    "placeholder",
    "none",
    "null",
})

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on", "y", "t"})
_FALSY_VALUES = frozenset({"0", "false", "no", "off", "n", "f", ""})


def _parse_bool(raw: str | None, *, default: bool) -> bool:
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in _TRUTHY_VALUES:
        return True
    if normalized in _FALSY_VALUES:
        return False
    return default


def _normalize_optional_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    cleaned = raw.strip()
    return cleaned if cleaned else None


def _normalize_project_id(raw: str | None) -> str | None:
    cleaned = _normalize_optional_str(raw)
    if cleaned is None or cleaned.lower() in _PLACEHOLDER_PROJECT_VALUES:
        return None
    return cleaned


def _resolve_enable_tracing(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    if "ENABLE_TRACING" in env:
        return _parse_bool(env.get("ENABLE_TRACING"), default=True)
    if "OTEL_EXPORT_TO_CLOUD" in env:
        return _parse_bool(env.get("OTEL_EXPORT_TO_CLOUD"), default=True)
    return True


def _resolve_port(environ: Mapping[str, str] | None = None) -> int:
    env = os.environ if environ is None else environ
    raw = env.get("PORT")
    if raw is None or not raw.strip():
        return DEFAULT_PORT
    try:
        parsed = int(raw.strip())
    except ValueError:
        return DEFAULT_PORT
    if 1 <= parsed <= 65535:
        return parsed
    return DEFAULT_PORT


class AppSettings(BaseModel):
    """Runtime settings resolved dynamically from environment variables."""

    model_config = ConfigDict(extra="forbid")

    app_name: str = Field(
        default_factory=lambda: _normalize_optional_str(os.getenv("THRILLED_APP_NAME"))
        or DEFAULT_APP_NAME
    )
    google_genai_use_vertexai: bool = Field(
        default_factory=lambda: _parse_bool(
            os.getenv("GOOGLE_GENAI_USE_VERTEXAI"), default=False
        )
    )
    google_cloud_project: str | None = Field(
        default_factory=lambda: _normalize_project_id(os.getenv("GOOGLE_CLOUD_PROJECT"))
    )
    google_cloud_location: str = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("GOOGLE_CLOUD_LOCATION")
        )
        or DEFAULT_GCP_LOCATION
    )
    gemini_api_key_secret_id: str = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("GEMINI_API_KEY_SECRET_ID")
        )
        or DEFAULT_SECRET_ID
    )
    enable_cloud_dlp: bool = Field(
        default_factory=lambda: _parse_bool(
            os.getenv("ENABLE_CLOUD_DLP"), default=False
        )
    )
    enable_model_armor: bool = Field(
        default_factory=lambda: _parse_bool(
            os.getenv("ENABLE_MODEL_ARMOR"), default=False
        )
    )
    model_armor_prompt_template: str | None = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("MODEL_ARMOR_PROMPT_TEMPLATE")
        )
    )
    model_armor_response_template: str | None = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("MODEL_ARMOR_RESPONSE_TEMPLATE")
        )
    )
    enable_tracing: bool = Field(default_factory=_resolve_enable_tracing)
    data_dir: Path = Field(
        default_factory=lambda: Path(
            _normalize_optional_str(os.getenv("THRILLED_DATA_DIR"))
            or str(DEFAULT_DATA_DIR)
        )
    )
    session_service_uri: str | None = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("SESSION_SERVICE_URI")
        )
    )
    use_vertex_memory_bank: bool = Field(
        default_factory=lambda: _parse_bool(
            os.getenv("THRILLED_USE_VERTEX_MEMORY_BANK"), default=False
        )
    )
    agent_engine_id: str | None = Field(
        default_factory=lambda: _normalize_optional_str(os.getenv("AGENT_ENGINE_ID"))
    )
    flash_model: str = Field(
        default_factory=lambda: _normalize_optional_str(
            os.getenv("THRILLED_FLASH_MODEL")
        )
        or DEFAULT_FLASH_MODEL
    )
    pro_model: str = Field(
        default_factory=lambda: _normalize_optional_str(os.getenv("THRILLED_PRO_MODEL"))
        or DEFAULT_PRO_MODEL
    )
    port: int = Field(default_factory=_resolve_port, ge=1, le=65535)

    @field_validator("google_cloud_project", mode="before")
    @classmethod
    def _validate_project_id(cls, value: str | None) -> str | None:
        return _normalize_project_id(value)

    @field_validator(
        "session_service_uri",
        "agent_engine_id",
        "model_armor_prompt_template",
        "model_armor_response_template",
        mode="before",
    )
    @classmethod
    def _validate_optional_str(cls, value: str | None) -> str | None:
        return _normalize_optional_str(value)

    @field_validator("data_dir", mode="before")
    @classmethod
    def _validate_data_dir(cls, value: str | Path) -> Path:
        if isinstance(value, Path):
            return value
        cleaned = str(value).strip()
        return Path(cleaned) if cleaned else DEFAULT_DATA_DIR

    @property
    def otel_export_to_cloud(self) -> bool:
        """Whether OpenTelemetry spans should also export to Google Cloud Trace."""
        if "OTEL_EXPORT_TO_CLOUD" in os.environ:
            return _parse_bool(os.getenv("OTEL_EXPORT_TO_CLOUD"), default=False)
        return bool(
            self.enable_tracing
            and self.google_cloud_project
            and self.google_genai_use_vertexai
        )

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> AppSettings:
        """Construct ``AppSettings`` from a provided mapping or live ``os.environ``."""
        if environ is None:
            return cls()
        return cls(
            app_name=_normalize_optional_str(environ.get("THRILLED_APP_NAME"))
            or DEFAULT_APP_NAME,
            google_genai_use_vertexai=_parse_bool(
                environ.get("GOOGLE_GENAI_USE_VERTEXAI"), default=False
            ),
            google_cloud_project=_normalize_project_id(
                environ.get("GOOGLE_CLOUD_PROJECT")
            ),
            google_cloud_location=_normalize_optional_str(
                environ.get("GOOGLE_CLOUD_LOCATION")
            )
            or DEFAULT_GCP_LOCATION,
            gemini_api_key_secret_id=_normalize_optional_str(
                environ.get("GEMINI_API_KEY_SECRET_ID")
            )
            or DEFAULT_SECRET_ID,
            enable_cloud_dlp=_parse_bool(environ.get("ENABLE_CLOUD_DLP"), default=False),
            enable_model_armor=_parse_bool(
                environ.get("ENABLE_MODEL_ARMOR"), default=False
            ),
            model_armor_prompt_template=_normalize_optional_str(
                environ.get("MODEL_ARMOR_PROMPT_TEMPLATE")
            ),
            model_armor_response_template=_normalize_optional_str(
                environ.get("MODEL_ARMOR_RESPONSE_TEMPLATE")
            ),
            enable_tracing=_resolve_enable_tracing(environ),
            data_dir=Path(
                _normalize_optional_str(environ.get("THRILLED_DATA_DIR"))
                or str(DEFAULT_DATA_DIR)
            ),
            session_service_uri=_normalize_optional_str(
                environ.get("SESSION_SERVICE_URI")
            ),
            use_vertex_memory_bank=_parse_bool(
                environ.get("THRILLED_USE_VERTEX_MEMORY_BANK"), default=False
            ),
            agent_engine_id=_normalize_optional_str(environ.get("AGENT_ENGINE_ID")),
            flash_model=_normalize_optional_str(environ.get("THRILLED_FLASH_MODEL"))
            or DEFAULT_FLASH_MODEL,
            pro_model=_normalize_optional_str(environ.get("THRILLED_PRO_MODEL"))
            or DEFAULT_PRO_MODEL,
            port=_resolve_port(environ),
        )


def reset_settings_cache() -> None:
    """Cache-reset hook retained for test fixture compatibility."""
    return None


def get_settings() -> AppSettings:
    """Return live environment-backed application settings."""
    return AppSettings.from_env()
