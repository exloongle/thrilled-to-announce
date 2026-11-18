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
"""Opaque-box tri-mode pytest harness for Thrilled to Announce E2E suites."""

from __future__ import annotations

import asyncio
import importlib
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
import pytest
import pytest_asyncio

E2EMode = Literal["real_asgi", "live_http", "selftest_stub"]
REAL_REPO_ROOT: Path = Path(__file__).resolve().parents[2]

_NO_RELOAD_PREFIXES: tuple[str, ...] = (
    "thrilled_to_announce.schemas",
    "thrilled_to_announce.observability.tracing",
    "thrilled_to_announce.observability.logging",
)


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    """Configure zero-warning asyncio loop scope and register tier1..tier4 markers."""
    if not config.getini("asyncio_default_fixture_loop_scope"):
        config._inicache["asyncio_default_fixture_loop_scope"] = "function"
    for tier_num, summary in (
        (1, "Tier 1 isolated happy-path feature coverage tests (F1-F12)"),
        (2, "Tier 2 boundary, corner, and negative robustness tests (F1-F12)"),
        (3, "Tier 3 cross-feature pairwise interaction tests"),
        (4, "Tier 4 multi-turn real-world application scenario journeys"),
    ):
        config.addinivalue_line("markers", f"tier{tier_num}: {summary}")


def _resolve_e2e_mode() -> E2EMode:
    """Resolve active E2E execution mode from environment variables."""
    if os.getenv("THRILLED_E2E_SELFTEST_STUB", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }:
        return "selftest_stub"
    if os.getenv("THRILLED_E2E_BASE_URL", "").strip():
        return "live_http"
    return "real_asgi"


def _load_fresh_real_app() -> Any:
    """Lazily import or reload real server.main:app after per-test env setup."""
    already_loaded = "server.main" in sys.modules
    for mod_name, mod in list(sys.modules.items()):
        if mod is None or not (
            mod_name.startswith("thrilled_to_announce") or mod_name.startswith("server")
        ):
            continue
        for fn_name in ("get_settings", "get_default_scrubber"):
            fn = getattr(mod, fn_name, None)
            cache_clear = getattr(fn, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()
        for clear_name in ("clear_telemetry_logs", "clear_recorded_spans"):
            clear_fn = getattr(mod, clear_name, None)
            if callable(clear_fn):
                clear_fn()

    try:
        if not already_loaded:
            server_main = importlib.import_module("server.main")
        else:
            server_main = sys.modules["server.main"]
            create_app_fn = getattr(server_main, "create_app", None)
            if not callable(create_app_fn):
                reload_targets = [
                    name
                    for name in list(sys.modules.keys())
                    if (
                        name.startswith("thrilled_to_announce.")
                        or name == "server.main"
                    )
                    and not name.startswith(_NO_RELOAD_PREFIXES)
                ]
                reload_targets.sort(
                    key=lambda name: (name == "server.main", -name.count("."))
                )
                for mod_name in reload_targets:
                    mod_obj = sys.modules.get(mod_name)
                    if mod_obj is not None:
                        importlib.reload(mod_obj)
                server_main = importlib.import_module("server.main")

        factory = getattr(server_main, "create_app", None)
        app = factory() if callable(factory) else getattr(server_main, "app", None)
        if app is None:
            pytest.fail(
                "Real module 'server.main' was imported but exposes neither "
                "'create_app()' nor 'app'."
            )
        return app
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, pytest.fail.Exception):
            raise exc
        pytest.fail(
            f"Failed to load real application 'server.main:app' in default E2E mode "
            f"(missing M4 implementation or broken import): {exc!r}"
        )


@dataclass(frozen=True)
class SqliteDataDirInspector:
    """Opaque-box inspector for SQLite persistence files inside THRILLED_DATA_DIR."""

    data_dir: Path
    is_remote_http: bool

    def _discover_sqlite_files(self) -> list[Path]:
        if not self.data_dir.exists():
            return []
        candidates: list[Path] = []
        for path in sorted(self.data_dir.rglob("*")):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            suffix = path.suffix.lower()
            if suffix in {".sqlite3", ".sqlite", ".db"}:
                candidates.append(path)
        return candidates

    async def wait_and_list_db_files(self, timeout_s: float = 1.2) -> list[Path]:
        """Poll until at least one non-empty SQLite file is flushed to disk."""
        if self.is_remote_http:
            return []
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            found = self._discover_sqlite_files()
            if found:
                return found
            await asyncio.sleep(0.03)
        return self._discover_sqlite_files()

    def _read_all_tables_sync(self) -> str:
        chunks: list[str] = []
        for sqlite_file in self._discover_sqlite_files():
            try:
                uri = f"file:{sqlite_file}?mode=ro"
                with closing(sqlite3.connect(uri, uri=True, timeout=2.0)) as conn:
                    tables = [
                        row[0]
                        for row in conn.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    ]
                    for table_name in tables:
                        safe_table = str(table_name).replace('"', '""')
                        rows = conn.execute(
                            f'SELECT * FROM "{safe_table}"'
                        ).fetchall()
                        for row in rows:
                            chunks.append(
                                " | ".join(
                                    "" if col is None else str(col) for col in row
                                )
                            )
            except sqlite3.Error:
                continue
        return "\n".join(chunks)

    async def dump_all_tables_text(
        self,
        expected_substring: str | None = None,
        timeout_s: float = 1.2,
    ) -> str:
        """Read all user tables across SQLite files in data_dir with bounded sync wait."""
        if self.is_remote_http:
            return ""
        deadline = time.monotonic() + timeout_s
        last_dump = ""
        while time.monotonic() < deadline:
            last_dump = self._read_all_tables_sync()
            if last_dump and (
                expected_substring is None or expected_substring in last_dump
            ):
                return last_dump
            await asyncio.sleep(0.03)
        return self._read_all_tables_sync()


@pytest.fixture(scope="session")
def real_repo_root() -> Path:
    """Return the actual workspace repository root path."""
    return REAL_REPO_ROOT


@pytest.fixture(scope="session")
def _selftest_overlay_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialize a clean self-test repository overlay only in selftest mode."""
    target = tmp_path_factory.mktemp("selftest_repo")
    stub_mod = importlib.import_module("tests.e2e._reference_stub")
    return stub_mod.materialize_selftest_repo_overlay(
        target_dir=target,
        real_repo_root=REAL_REPO_ROOT,
    )


@pytest.fixture
def e2e_mode() -> E2EMode:
    """Return current E2E execution mode."""
    return _resolve_e2e_mode()


@pytest.fixture
def is_live_remote(e2e_mode: E2EMode) -> bool:
    """Return True when running against an external THRILLED_E2E_BASE_URL server."""
    return e2e_mode == "live_http"


@pytest.fixture
def project_root(
    e2e_mode: E2EMode,
    request: pytest.FixtureRequest,
) -> Path:
    """Return real repository root in default/live modes or overlay in selftest mode."""
    if e2e_mode == "selftest_stub":
        return request.getfixturevalue("_selftest_overlay_root")
    return REAL_REPO_ROOT


@pytest.fixture
def repo_root(project_root: Path) -> Path:
    """Alias for project_root."""
    return project_root


@pytest.fixture
def data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    e2e_mode: E2EMode,
) -> Path:
    """Provide a fresh isolated THRILLED_DATA_DIR for every test."""
    isolated = tmp_path / "thrilled_data"
    isolated.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("THRILLED_DATA_DIR", str(isolated))
    monkeypatch.delenv("SESSION_SERVICE_URI", raising=False)
    if e2e_mode != "live_http" and os.getenv("THRILLED_E2E_LIVE_VERTEX", "").strip() != "1":
        monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")
        monkeypatch.setenv("ENABLE_CLOUD_DLP", "FALSE")
        monkeypatch.setenv("OTEL_EXPORT_TO_CLOUD", "FALSE")
        monkeypatch.setenv("ENABLE_TRACING", "TRUE")
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return isolated


@pytest.fixture
def unique_user_id() -> str:
    """Generate an isolated user_id per test (safe for local and remote modes)."""
    return f"user_e2e_{uuid.uuid4().hex[:16]}"


@pytest.fixture
def user_id_factory() -> Callable[[str], str]:
    """Return a factory creating distinct per-persona isolated user_ids."""

    def _make(prefix: str = "persona") -> str:
        clean = "".join(ch for ch in prefix if ch.isalnum() or ch == "_")[:24]
        return f"u_{clean}_{uuid.uuid4().hex[:12]}"

    return _make


@pytest_asyncio.fixture(loop_scope="function")
async def api_client(
    e2e_mode: E2EMode,
    data_dir: Path,
) -> AsyncIterator[httpx.AsyncClient]:
    """Yield an opaque-box AsyncClient wired to real ASGI, live URL, or selftest app."""
    if e2e_mode == "live_http":
        base_url = os.environ["THRILLED_E2E_BASE_URL"].strip().rstrip("/")
        token = os.getenv("THRILLED_E2E_BEARER_TOKEN", "").strip()
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        ) as live_client:
            yield live_client
        return

    if e2e_mode == "selftest_stub":
        stub_mod = importlib.import_module("tests.e2e._reference_stub")
        app = stub_mod.create_reference_stub_app(
            data_dir=data_dir,
            real_repo_root=REAL_REPO_ROOT,
        )
    else:
        app = _load_fresh_real_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            timeout=30.0,
            trust_env=False,
        ) as asgi_client:
            yield asgi_client


@pytest_asyncio.fixture(loop_scope="function")
async def e2e_client(api_client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    """Primary opaque-box HTTP client alias used across Tiers 1-4."""
    yield api_client


@pytest_asyncio.fixture(loop_scope="function")
async def client(api_client: httpx.AsyncClient) -> AsyncIterator[httpx.AsyncClient]:
    """Secondary HTTP client alias."""
    yield api_client


@pytest.fixture
def sqlite_inspector(e2e_mode: E2EMode, data_dir: Path) -> SqliteDataDirInspector:
    """Return an opaque-box SQLite file inspector bound to the test's data_dir."""
    return SqliteDataDirInspector(
        data_dir=data_dir,
        is_remote_http=(e2e_mode == "live_http"),
    )


@pytest.fixture
def run_eval_cli(
    e2e_mode: E2EMode,
    project_root: Path,
    data_dir: Path,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Invoke `python -m eval.run_eval` in an isolated subprocess."""

    def _invoke(*extra_args: str) -> subprocess.CompletedProcess[str]:
        eval_file = project_root / "eval" / "run_eval.py"
        if e2e_mode != "selftest_stub" and not eval_file.is_file():
            pytest.fail(
                f"Evaluation CLI entry point missing at {eval_file} in {e2e_mode} mode."
            )
        eval_data_dir = data_dir / f"eval_cli_{uuid.uuid4().hex[:8]}"
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["THRILLED_DATA_DIR"] = str(eval_data_dir)
        env["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
        env["ENABLE_CLOUD_DLP"] = "FALSE"
        env["OTEL_EXPORT_TO_CLOUD"] = "FALSE"
        env.pop("GOOGLE_CLOUD_PROJECT", None)
        env.pop("GEMINI_API_KEY", None)
        if e2e_mode == "selftest_stub":
            env["PYTHONPATH"] = f"{project_root}{os.pathsep}{REAL_REPO_ROOT}"
        else:
            env["PYTHONPATH"] = str(project_root)
        return subprocess.run(
            [sys.executable, "-m", "eval.run_eval", *extra_args],
            cwd=str(project_root),
            env=env,
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )

    return _invoke


@pytest.fixture
def eval_runner(
    run_eval_cli: Callable[..., subprocess.CompletedProcess[str]],
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Alias for run_eval_cli."""
    return run_eval_cli
