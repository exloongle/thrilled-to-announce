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
"""Secure Secret Manager credential resolver with IAM ADC awareness and in-memory cache."""

from __future__ import annotations

import os
from typing import Any, Literal

from google.cloud import secretmanager
from google.cloud.secretmanager import SecretManagerServiceClient

from thrilled_to_announce.config.settings import AppSettings, get_settings

_PLACEHOLDER_SECRET_VALUES = frozenset({
    "",
    "your_api_key",
    "<your_api_key>",
    "your-api-key",
    "changeme",
    "none",
    "null",
})


class SecretManagerResolver:
    """Resolves runtime secrets from Google Cloud Secret Manager without persisting or logging values."""

    def __init__(
        self,
        project_id: str | None = None,
        client: Any | None = None,
    ) -> None:
        self._explicit_project_id = (
            project_id.strip() if project_id and project_id.strip() else None
        )
        self._client = client
        self._cache: dict[tuple[str, str, str], str] = {}

    def __repr__(self) -> str:
        return (
            f"SecretManagerResolver(project_id={self.project_id!r}, "
            f"cached_entries={len(self._cache)})"
        )

    @property
    def project_id(self) -> str | None:
        """Effective GCP project ID from explicit override or live ``AppSettings``."""
        if (
            self._explicit_project_id
            and self._explicit_project_id.lower() != "your_project_id"
        ):
            return self._explicit_project_id
        return get_settings().google_cloud_project

    @property
    def uses_vertex_adc(self) -> bool:
        """True when Vertex AI IAM Application Default Credentials mode is active."""
        return get_settings().google_genai_use_vertexai

    @property
    def cached_secret_count(self) -> int:
        """Number of secrets held in memory without exposing raw secret payloads."""
        return len(self._cache)

    def clear_cache(self) -> None:
        """Purge all in-memory cached secret payloads."""
        self._cache.clear()

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    @staticmethod
    def _build_secret_version_name(
        project_id: str | None, secret_id: str, version: str
    ) -> str | None:
        cleaned_secret = secret_id.strip()
        if not cleaned_secret:
            return None
        if cleaned_secret.startswith("projects/"):
            if "/versions/" in cleaned_secret:
                return cleaned_secret
            return f"{cleaned_secret.rstrip('/')}/versions/{version}"
        if not project_id:
            return None
        return SecretManagerServiceClient.secret_version_path(
            project_id, cleaned_secret, version
        )

    @staticmethod
    def _read_env_fallback(env_var_name: str | None) -> str | None:
        if not env_var_name:
            return None
        candidates = [env_var_name]
        if env_var_name == "GEMINI_API_KEY":
            candidates.append("GOOGLE_API_KEY")
        for var_name in candidates:
            raw = os.getenv(var_name)
            if raw is None:
                continue
            cleaned = raw.strip()
            if cleaned and cleaned.lower() not in _PLACEHOLDER_SECRET_VALUES:
                return cleaned
        return None

    def resolve_secret(
        self,
        secret_id: str,
        version: str = "latest",
        *,
        env_fallback_var: str | None = None,
    ) -> str | None:
        """Resolve a secret payload via ``SecretManagerServiceClient.access_secret_version``."""
        cleaned_secret = secret_id.strip()
        if not cleaned_secret:
            return self._read_env_fallback(env_fallback_var)

        effective_project = self.project_id
        cache_key = (effective_project or "", cleaned_secret, version)
        if cache_key in self._cache:
            return self._cache[cache_key]

        resource_name = self._build_secret_version_name(
            effective_project, cleaned_secret, version
        )
        if resource_name is not None:
            try:
                client = self._get_client()
                response = client.access_secret_version(request={"name": resource_name})
                payload_data = getattr(getattr(response, "payload", None), "data", b"")
                decoded = (
                    payload_data.decode("utf-8").strip()
                    if isinstance(payload_data, (bytes, bytearray))
                    else str(payload_data).strip()
                )
                if decoded and decoded.lower() not in _PLACEHOLDER_SECRET_VALUES:
                    self._cache[cache_key] = decoded
                    return decoded
            except Exception:
                pass

        return self._read_env_fallback(env_fallback_var)

    def get_gemini_api_key(
        self,
        *,
        settings: AppSettings | None = None,
        force_secret_lookup: bool = False,
    ) -> str | None:
        """Resolve the Gemini API key when not running under Vertex AI IAM ADC."""
        effective_settings = settings or get_settings()
        if effective_settings.google_genai_use_vertexai and not force_secret_lookup:
            return None
        return self.resolve_secret(
            effective_settings.gemini_api_key_secret_id,
            version="latest",
            env_fallback_var="GEMINI_API_KEY",
        )

    def resolve_auth_mode(
        self, *, settings: AppSettings | None = None
    ) -> Literal["vertex_adc", "secret_manager", "env_fallback", "unconfigured"]:
        """Report the active authentication mechanism without exposing secret values."""
        effective_settings = settings or get_settings()
        if effective_settings.google_genai_use_vertexai:
            return "vertex_adc"
        cache_probe = (
            self.project_id or "",
            effective_settings.gemini_api_key_secret_id.strip(),
            "latest",
        )
        resolved = self.get_gemini_api_key(settings=effective_settings)
        if resolved is None:
            return "unconfigured"
        if cache_probe in self._cache:
            return "secret_manager"
        return "env_fallback"


_DEFAULT_RESOLVER: SecretManagerResolver | None = None


def get_secret_resolver() -> SecretManagerResolver:
    """Return the shared in-memory ``SecretManagerResolver`` instance."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is None:
        _DEFAULT_RESOLVER = SecretManagerResolver()
    return _DEFAULT_RESOLVER


def reset_secret_resolver() -> None:
    """Clear and reset the shared ``SecretManagerResolver`` singleton."""
    global _DEFAULT_RESOLVER
    if _DEFAULT_RESOLVER is not None:
        _DEFAULT_RESOLVER.clear_cache()
    _DEFAULT_RESOLVER = None
