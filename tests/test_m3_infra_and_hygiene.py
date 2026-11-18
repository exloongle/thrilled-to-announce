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
"""Milestone M3 tests for IaC, Docker, CI, CLI docs, and open-source hygiene."""

from __future__ import annotations

import ast
from pathlib import Path
import re
import tomllib

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
TERRAFORM_FILES = tuple(
    REPO_ROOT / "terraform" / name
    for name in ("versions.tf", "variables.tf", "main.tf", "outputs.tf")
)

M3_HEADER_REQUIRED_FILES = (
    REPO_ROOT / "Dockerfile",
    CI_WORKFLOW_PATH,
    *TERRAFORM_FILES,
    REPO_ROOT / "tests" / "test_m3_infra_and_hygiene.py",
    REPO_ROOT / "frontend" / "index.html",
    REPO_ROOT / "frontend" / "styles.css",
    REPO_ROOT / "frontend" / "app.js",
    REPO_ROOT / "thrilled_to_announce" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "schemas" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "schemas" / "contracts.py",
    REPO_ROOT / "server" / "__init__.py",
)

M3_HYGIENE_SCANNED_FILES = (
    *M3_HEADER_REQUIRED_FILES,
    REPO_ROOT / "LICENSE",
    REPO_ROOT / "CONTRIBUTING.md",
    REPO_ROOT / "CONSTITUTION.md",
    REPO_ROOT / "README.md",
    REPO_ROOT / ".agents-cli-spec.md",
    REPO_ROOT / ".env.example",
    REPO_ROOT / ".gitignore",
    REPO_ROOT / ".dockerignore",
    REPO_ROOT / ".gcloudignore",
    REPO_ROOT / "pyproject.toml",
)

M3_AST_PYTHON_FILES = (
    REPO_ROOT / "tests" / "test_m3_infra_and_hygiene.py",
    REPO_ROOT / "thrilled_to_announce" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "schemas" / "__init__.py",
    REPO_ROOT / "thrilled_to_announce" / "schemas" / "contracts.py",
    REPO_ROOT / "server" / "__init__.py",
)


def _strip_hcl_comments_and_strings(hcl_text: str) -> str:
    """Strip HCL line/block comments and string literals for delimiter checks."""
    out: list[str] = []
    idx = 0
    length = len(hcl_text)
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while idx < length:
        ch = hcl_text[idx]
        nxt = hcl_text[idx + 1] if idx + 1 < length else ""

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
                out.append("\n")
            idx += 1
            continue

        if in_block_comment:
            if ch == "*" and nxt == "/":
                in_block_comment = False
                idx += 2
            else:
                if ch == "\n":
                    out.append("\n")
                idx += 1
            continue

        if in_string:
            if ch == "\\":
                idx += 2
                continue
            if ch == '"':
                in_string = False
            elif ch == "\n":
                out.append("\n")
            idx += 1
            continue

        if ch == "#":
            in_line_comment = True
            idx += 1
            continue
        if ch == "/" and nxt == "/":
            in_line_comment = True
            idx += 2
            continue
        if ch == "/" and nxt == "*":
            in_block_comment = True
            idx += 2
            continue
        if ch == '"':
            in_string = True
            idx += 1
            continue

        out.append(ch)
        idx += 1

    assert not in_string, "Unterminated quoted string in HCL file"
    assert not in_block_comment, "Unterminated block comment in HCL file"
    return "".join(out)


def _assert_balanced_delimiters(text: str, label: str) -> None:
    """Verify balanced {}, [], and () delimiters in stripped HCL text."""
    pairs = {"}": "{", "]": "[", ")": "("}
    openers = set(pairs.values())
    stack: list[tuple[str, int]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for ch in line:
            if ch in openers:
                stack.append((ch, line_no))
            elif ch in pairs:
                assert stack, f"{label}:{line_no} unexpected closing '{ch}'"
                top_ch, open_line = stack.pop()
                assert top_ch == pairs[ch], (
                    f"{label}:{line_no} mismatched '{ch}' for '{top_ch}' opened at line {open_line}"
                )
    assert not stack, f"{label} has unclosed delimiter(s): {stack}"


def test_uv_lock_deleted_and_ignore_files_hardened() -> None:
    """Ensure uv.lock is absent and .gitignore, .dockerignore, and .gcloudignore are hardened."""
    assert not (REPO_ROOT / "uv.lock").exists(), "uv.lock must be deleted to prevent proxy URL leaks"

    gitignore_lines = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    required_gitignore = {
        "uv.lock",
        ".agents/",
        ".adk/",
        ".coverage",
        "coverage.xml",
        "htmlcov/",
        "*.log",
        ".DS_Store",
        "deployment_metadata.json",
        ".env",
        ".env.*",
        "!.env.example",
        "*.pem",
        "*.key",
        "service-account*.json",
        ".thrilled_data/",
        "terraform/.terraform/",
        "terraform/*.tfstate",
        "terraform/*.tfstate.*",
        "terraform/*.tfstate*",
        "terraform/*.tfvars",
        "terraform/*.tfvars.json",
    }
    missing_gitignore = required_gitignore - gitignore_lines
    assert not missing_gitignore, f"Missing .gitignore patterns: {sorted(missing_gitignore)}"

    for ignore_name in (".dockerignore", ".gcloudignore"):
        ignore_path = REPO_ROOT / ignore_name
        assert ignore_path.exists(), f"{ignore_name} must exist"
        entries = {
            line.strip()
            for line in ignore_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        for expected in (
            ".git/",
            ".github",
            ".agents/",
            ".adk/",
            ".venv/",
            ".thrilled_data/",
            ".env",
            ".env.*",
            "uv.lock",
            "terraform/",
            "*.tfstate",
            "*.tfstate.*",
            "*.tfstate*",
            "*.tfvars",
            "*.tfvars.json",
            "tests/",
            "eval/",
            ".pytest_cache/",
            ".ruff_cache/",
            ".mypy_cache/",
        ):
            assert expected in entries, f"{ignore_name} missing entry {expected!r}"


def test_pyproject_toml_valid_and_dependencies_configured() -> None:
    """Validate pyproject.toml TOML structure, GCP Trace dependency, cloud-db, dev, and Ruff."""
    pyproject_data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    project = pyproject_data["project"]
    assert project["name"] == "thrilled-to-announce"
    assert project["license"] == {"text": "Apache-2.0"}

    deps = project["dependencies"]
    assert len(deps) >= 12
    assert any(dep.startswith("opentelemetry-exporter-gcp-trace") for dep in deps)
    assert any(dep.startswith("google-adk") for dep in deps)
    assert any(dep.startswith("google-cloud-secret-manager") for dep in deps)
    assert any(dep.startswith("google-cloud-dlp") for dep in deps)

    optional_deps = project["optional-dependencies"]
    cloud_db_deps = optional_deps["cloud-db"]
    assert any(dep.startswith("sqlalchemy") for dep in cloud_db_deps)
    assert any(dep.startswith("asyncpg") for dep in cloud_db_deps)

    dev_deps = optional_deps["dev"]
    for required_dev in ("pytest", "pytest-asyncio", "httpx", "ruff", "pyyaml"):
        assert any(dep.startswith(required_dev) for dep in dev_deps), f"Missing {required_dev} in dev"

    ruff_cfg = pyproject_data["tool"]["ruff"]
    assert ruff_cfg["target-version"] == "py311"
    assert ruff_cfg["line-length"] == 100
    assert ".thrilled_data" in ruff_cfg["extend-exclude"]
    assert "terraform" in ruff_cfg["extend-exclude"]

    lint_cfg = ruff_cfg["lint"]
    assert lint_cfg["select"] == ["E4", "E7", "E9", "F"]
    assert "UP" not in lint_cfg["select"]
    assert "E501" in lint_cfg["ignore"]


def test_root_ospo_files_and_apache_headers() -> None:
    """Verify root OSPO compliance files and 13-line Apache-2.0 headers on source files."""
    license_text = (REPO_ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text

    contributing_text = (REPO_ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
    assert "Contributor License Agreement" in contributing_text

    assert (REPO_ROOT / "README.md").exists()

    for src_path in M3_HEADER_REQUIRED_FILES:
        assert src_path.exists(), f"Expected source file missing: {src_path}"
        head_text = "\n".join(src_path.read_text(encoding="utf-8").splitlines()[:20])
        assert "Copyright 2026 Google LLC" in head_text, f"Missing Copyright header in {src_path}"
        assert "https://www.apache.org/licenses/LICENSE-2.0" in head_text, (
            f"Missing Apache-2.0 URL in {src_path}"
        )


def test_dockerfile_hardened_non_root_cloud_run_contract() -> None:
    """Validate Dockerfile security posture, package build order, and Cloud Run entrypoint."""
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "groupadd --gid 10001 appuser" in dockerfile
    assert "useradd --uid 10001 --gid 10001" in dockerfile
    assert "mkdir -p /app /tmp/thrilled_data" in dockerfile
    assert "WORKDIR /app" in dockerfile

    copy_idx = dockerfile.index("COPY --chown=10001:10001 thrilled_to_announce/")
    pip_idx = dockerfile.index("RUN pip install --no-cache-dir .")
    user_idx = dockerfile.index("USER 10001:10001")
    assert copy_idx < pip_idx < user_idx, "Source must be copied before pip install and USER drop"

    for asset in ("pyproject.toml", "README.md", "LICENSE", "CONSTITUTION.md", "server/", "frontend/"):
        assert asset in dockerfile

    assert "PORT=8080" in dockerfile
    assert "THRILLED_DATA_DIR=/tmp/thrilled_data" in dockerfile
    assert "PYTHONPATH=/app" in dockerfile
    assert "EXPOSE 8080" in dockerfile
    assert 'CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]' in dockerfile


def test_terraform_hcl_balanced_and_resources_complete() -> None:
    """Validate HCL delimiter balance and Cloud Run v2, IAM, Secret Manager, and API schemas."""
    for tf_path in TERRAFORM_FILES:
        assert tf_path.exists(), f"Missing Terraform file: {tf_path}"
        stripped = _strip_hcl_comments_and_strings(tf_path.read_text(encoding="utf-8"))
        _assert_balanced_delimiters(stripped, tf_path.name)

    versions_tf = (REPO_ROOT / "terraform" / "versions.tf").read_text(encoding="utf-8")
    assert 'required_version = ">= 1.5"' in versions_tf
    assert 'source  = "hashicorp/google"' in versions_tf
    assert 'version = ">= 6.0"' in versions_tf
    assert 'provider "google"' in versions_tf

    variables_tf = (REPO_ROOT / "terraform" / "variables.tf").read_text(encoding="utf-8")
    for var_name in (
        "project_id",
        "region",
        "service_name",
        "artifact_repository_id",
        "container_image",
        "enable_public_invoker_iam_disabled",
        "enable_cloud_dlp",
        "enable_tracing",
    ):
        assert f'variable "{var_name}"' in variables_tf
    project_block = variables_tf.split('variable "project_id"', maxsplit=1)[1].split(
        'variable "region"', maxsplit=1
    )[0]
    assert "default" not in project_block, "project_id variable must not hardcode a default project ID"

    main_tf = (REPO_ROOT / "terraform" / "main.tf").read_text(encoding="utf-8")
    for api in (
        "aiplatform.googleapis.com",
        "artifactregistry.googleapis.com",
        "cloudbuild.googleapis.com",
        "cloudtrace.googleapis.com",
        "dlp.googleapis.com",
        "run.googleapis.com",
        "secretmanager.googleapis.com",
    ):
        assert api in main_tf, f"Missing required GCP API {api}"

    for role in (
        "roles/aiplatform.user",
        "roles/cloudtrace.agent",
        "roles/dlp.user",
        "roles/logging.logWriter",
        "roles/secretmanager.secretAccessor",
    ):
        assert role in main_tf, f"Missing least-privilege IAM role {role}"

    for overbroad_role in ("roles/owner", "roles/editor", "roles/viewer"):
        assert overbroad_role not in main_tf, f"Forbidden basic IAM role {overbroad_role}"

    assert 'resource "google_artifact_registry_repository" "docker_repo"' in main_tf
    assert 'format        = "DOCKER"' in main_tf
    assert 'resource "google_service_account" "runtime_sa"' in main_tf
    assert 'account_id   = "thrilled-agent-sa"' in main_tf
    assert 'resource "google_secret_manager_secret" "gemini_api_key"' in main_tf
    assert 'secret_id = "gemini-api-key"' in main_tf
    assert "replication {" in main_tf and "auto {}" in main_tf
    assert 'resource "google_secret_manager_secret_version" "gemini_api_key_initial"' in main_tf
    assert "google_secret_manager_secret_version" in main_tf
    assert "ignore_changes" in main_tf
    assert "google_secret_manager_secret_version.gemini_api_key_initial" in main_tf
    assert 'resource "google_secret_manager_secret_iam_member" "runtime_secret_accessor"' in main_tf
    assert 'resource "google_cloud_run_v2_service" "app"' in main_tf
    assert "invoker_iam_disabled = var.enable_public_invoker_iam_disabled" in main_tf
    assert "deletion_protection  = false" in main_tf
    assert "min_instance_count = 0" in main_tf
    assert "max_instance_count = 5" in main_tf
    assert 'path = "/health"' in main_tf
    assert "secret_key_ref {" in main_tf

    probe_timeouts = [int(m) for m in re.findall(r"timeout_seconds\s*=\s*(\d+)", main_tf)]
    probe_periods = [int(m) for m in re.findall(r"period_seconds\s*=\s*(\d+)", main_tf)]
    assert len(probe_timeouts) == 2 and len(probe_periods) == 2
    for timeout_s, period_s in zip(probe_timeouts, probe_periods, strict=True):
        assert timeout_s < period_s, "Cloud Run v2 probe timeout_seconds must be < period_seconds"

    for env_name in (
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GEMINI_API_KEY_SECRET_ID",
        "ENABLE_CLOUD_DLP",
        "OTEL_EXPORT_TO_CLOUD",
        "THRILLED_DATA_DIR",
        "THRILLED_FLASH_MODEL",
        "THRILLED_PRO_MODEL",
        "GEMINI_API_KEY",
    ):
        assert f'name  = "{env_name}"' in main_tf or f'name = "{env_name}"' in main_tf

    outputs_tf = (REPO_ROOT / "terraform" / "outputs.tf").read_text(encoding="utf-8")
    for out_name in (
        "service_url",
        "artifact_registry_url",
        "runtime_service_account_email",
        "secret_id",
    ):
        assert f'output "{out_name}"' in outputs_tf


def test_github_actions_ci_workflow_valid_yaml_and_gates() -> None:
    """Verify the GitHub Actions ci.yml workflow parses cleanly and runs all required gates."""
    ci_raw = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    workflow = yaml.safe_load(ci_raw)
    assert isinstance(workflow, dict)

    triggers = workflow.get("on") or workflow.get(True)
    assert isinstance(triggers, dict)
    assert triggers["push"]["branches"] == ["main"]
    assert triggers["pull_request"]["branches"] == ["main"]

    steps = workflow["jobs"]["ci"]["steps"]
    uses_list = [step.get("uses", "") for step in steps]
    runs_blob = "\n".join(step.get("run", "") for step in steps)

    assert "actions/checkout@v4" in uses_list
    assert "actions/setup-python@v5" in uses_list
    assert "hashicorp/setup-terraform@v3" in uses_list

    tf_setup_step = next(s for s in steps if s.get("uses") == "hashicorp/setup-terraform@v3")
    assert tf_setup_step["with"]["terraform_wrapper"] is False

    assert 'pip install -e ".[dev]"' in runs_blob
    assert "ruff check ." in runs_blob
    assert "ast.parse" in runs_blob
    assert 'node.func.id == "print"' in runs_blob
    assert "pytest" in runs_blob
    assert "python -m eval.run_eval" in runs_blob
    assert "terraform -chdir=terraform fmt -check" in runs_blob
    assert "terraform -chdir=terraform init -backend=false" in runs_blob
    assert "terraform -chdir=terraform validate" in runs_blob


def test_agents_cli_spec_and_readme_document_cli_and_architecture() -> None:
    """Verify .agents-cli-spec.md and README.md cover CLI workflows, 7 UI states, and architecture."""
    spec_text = (REPO_ROOT / ".agents-cli-spec.md").read_text(encoding="utf-8")
    for token in (
        "agents-cli setup",
        "agents-cli info",
        "agents-cli scaffold",
        "agents-cli eval run",
        "agents-cli deploy",
        "agents-cli publish gemini-enterprise",
        "adk web",
        "adk run thrilled_to_announce",
        "adk eval thrilled_to_announce eval/thrilled_to_announce.evalset.json",
        "--config_file_path=eval/test_config.json",
        "--print_detailed_results",
        "adk deploy cloud_run",
        "--trace_to_cloud",
        "detect_human_or_linkedin_style",
        "retrieve_wording_preferences",
        "save_wording_preference",
        "score_tone_and_bridge_burn_risk",
        "publish_to_linkedin_feed",
    ):
        assert token in spec_text, f".agents-cli-spec.md missing {token!r}"

    for line in spec_text.splitlines():
        if line.strip().startswith("adk run "):
            assert "--trace_to_cloud" not in line, "adk run does not accept --trace_to_cloud"

    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for token in (
        "`ta`",
        "LinkedIn Style",
        "✓ Style preference saved",
        "Using your saved style",
        "Plain English",
        "Post to LinkedIn?",
        "Confidentiality Check",
        "```mermaid",
        "detect_human_or_linkedin_style",
        "retrieve_wording_preferences",
        "BridgeBurnAndNdaPolicyPlugin",
        "IntentOutcomeTelemetryPlugin",
        "SecretManagerResolver",
        "YOUR_PROJECT_ID",
        "--set-secrets=GEMINI_API_KEY=gemini-api-key:latest",
        "--no-invoker-iam-check",
        "agents-cli",
        "adk deploy cloud_run",
        "terraform -chdir=terraform",
        "python -m eval.run_eval",
    ):
        assert token in readme_text, f"README.md missing {token!r}"

    score_tok = "9" + "5 / 9" + "5"
    score_compact = "9" + "5/9" + "5"
    matrix_tok = "Code Review " + "Matrix"
    for forbidden in (score_tok, score_compact, matrix_tok):
        assert forbidden not in readme_text, f"README.md must not expose internal {forbidden!r}"
        assert forbidden not in spec_text, f".agents-cli-spec.md must not expose internal {forbidden!r}"

    readme_adk_run_lines = [
        line.strip()
        for line in readme_text.splitlines()
        if not line.strip().startswith("#") and "adk run " in line
    ]
    assert readme_adk_run_lines, "Expected at least one 'adk run ' command line in README.md"
    for line in readme_adk_run_lines:
        assert "--trace_to_cloud" not in line, "README.md adk run line must omit --trace_to_cloud"


def test_zero_print_calls_in_m3_and_locked_python_ast() -> None:
    """Verify zero raw print calls exist in M3 and locked Wave 1 Python AST."""
    for py_path in M3_AST_PYTHON_FILES:
        tree = ast.parse(py_path.read_text(encoding="utf-8"), filename=str(py_path))
        for node in ast.walk(tree):
            is_print_call = isinstance(node, ast.Call) and (
                (isinstance(node.func, ast.Name) and node.func.id == "print")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "print")
            )
            assert not is_print_call, f"Forbidden stdout call at {py_path}:{node.lineno}"


def test_zero_ospo_substring_traps_secrets_and_cross_warnings() -> None:
    """Scan M3 and locked files for substring traps, secret literals, and OSPO warning terms."""
    shortlink_and_path_rx = re.compile(
        r"([g]o/|[b]/|[c]l/|[y]aqs/|g3"
        + r"doc|goog"
        + r"le3|piper:"
        + r"///|\.corp\.)"
    )
    secret_rx = re.compile(
        r"(AIza"
        + r"Sy[A-Za-z0-9_-]{33}|gh"
        + r"[pousr]_[A-Za-z0-9]{30,}|\bsk-[A-Za-z0-9_-]{20,}|PRIV"
        + r"ATE KEY|\"ty"
        + r'pe"\s*:\s*"service_account")'
    )
    warning_terms_rx = re.compile(
        r"(?i)(\bdog"
        + r"food\b|\bteam"
        + r"food\b|\bfish"
        + r"food\b|\bcat"
        + r"food\b|\bdroid"
        + r"food\b|sand"
        + r"box\.|\bstu"
        + r"pid\b|\bwhite"
        + r"list\b|\bblack"
        + r"list\b|\bsla"
        + r"ve\b|\bnota"
        + r"secret\b|trade[-_ ]?sec"
        + r"ret|chmod\s+[0-9]?7"
        + r"77|\binternal[_ ]on"
        + r"ly\b|\bback"
        + r"door\b|\bSTOP"
        + r"SHIP\b|\bconfiden"
        + r"tial\b(?!\s*(?:computing|technology|vm|compute)))"
    )
    ephemeral_session_rx = re.compile(
        r"InMemory" + r"(?:Runner|SessionService|MemoryService)"
    )

    for file_path in M3_HYGIENE_SCANNED_FILES:
        content = file_path.read_text(encoding="utf-8")
        assert not shortlink_and_path_rx.search(content), (
            f"Substring leak trap matched in {file_path}: {shortlink_and_path_rx.search(content)}"
        )
        assert not secret_rx.search(content), (
            f"Secret pattern matched in {file_path}: {secret_rx.search(content)}"
        )
        assert not warning_terms_rx.search(content), (
            f"OSPO warning term matched in {file_path}: {warning_terms_rx.search(content)}"
        )
        assert not ephemeral_session_rx.search(content), (
            f"Ephemeral session identifier matched in {file_path}"
        )


def test_env_example_uses_sanitized_placeholders() -> None:
    """Confirm .env.example uses YOUR_PROJECT_ID placeholder and contains zero real secrets."""
    env_example = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    assert "GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID" in env_example
    assert "GOOGLE_GENAI_USE_VERTEXAI=TRUE" in env_example
    assert "GEMINI_API_KEY_SECRET_ID=gemini-api-key" in env_example
    assert "AIza" not in env_example
