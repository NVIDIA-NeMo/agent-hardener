# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set dotenv-load
set windows-shell := ["powershell.exe", "-NoLogo", "-NonInteractive", "-Command"]

# NOTE: Recipes using just's interpolation (double-brace syntax) must be inside
# raw/endraw blocks to prevent copier from interpreting them as template variables.
rm_rf := if os() == "windows" { "powershell -Command Remove-Item -Recurse -Force -ErrorAction SilentlyContinue" } else { "rm -rf" }

# Show available commands
default:
    @just --list

alias help := default

# Initialize git repo (idempotent — skips if .git exists)
[unix]
git-init:
    @bash scripts/init-local-git-repo.sh

[windows]
git-init:
    @powershell -ExecutionPolicy Bypass -File scripts/init-local-git-repo.ps1

# Add remote and push (idempotent — skips if origin exists)
[unix]
git-remote remote_url:
    {{ if remote_url == "" { error("No remote URL provided. Usage: just git-remote <url>") } else { "" } }}
    @[[ '{{ remote_url }}' =~ ^(https?://|ssh://|git@) ]] || { echo "error: remote URL must start with https://, ssh://, or git@" >&2; exit 1; }
    @bash scripts/configure-git-origin.sh '{{ remote_url }}'

[windows]
git-remote remote_url:
    {{ if remote_url == "" { error("No remote URL provided. Usage: just git-remote <url>") } else { "" } }}
    @if ('{{ remote_url }}' -notmatch '^(https?://|ssh://|git@)') { Write-Error 'error: remote URL must start with https://, ssh://, or git@'; exit 1 }
    @powershell -ExecutionPolicy Bypass -File scripts/configure-git-origin.ps1 '{{ remote_url }}'


# First-time setup: install dependencies, configure OpenShell, and create .env files
[unix]
setup:
    ./scripts/setup.sh

# Tear down the running Agent Hardener runtime (gateway preserved unless --stop-gateway)
[unix]
stop *args:
    ./scripts/stop-swarm.sh {{ args }}

# Install Python, sync dependencies, export requirements.txt, and set up pre-commit hooks
[unix]
install:
    @bash scripts/install-project.sh
    just garak-venv
    if command -v brew >/dev/null 2>&1; then brew list e2fsprogs >/dev/null 2>&1 || HOMEBREW_NO_AUTO_UPDATE=1 brew install e2fsprogs; fi

[windows]
install:
    @powershell -ExecutionPolicy Bypass -File scripts/install-project.ps1

# Generate dependency artifacts without installing hooks (internal helper).
# Inlines the `uv export` from `export-reqs` rather than calling `just export-reqs`:
# during `copier copy` this runs via `uvx --from rust-just just bootstrap-deps`, where
# a bare `just` is not reliably on PATH. Keep in sync with the `export-reqs` recipe.
[private]
bootstrap-deps:
    uv python install 3.11
    uv sync
    uv export --format requirements-txt --no-hashes --no-emit-project --output-file requirements.txt

# Install project dependencies in CI (frozen lockfile, no pre-commit hooks)
install-ci:
    #!/usr/bin/env bash
    set -euo pipefail
    uv python find 3.11 >/dev/null 2>&1 || uv python install 3.11
    uv sync --frozen

# Provision the dedicated garak venv (~/.agent-hardener/garak-venv) the agent_breaker attacker
# spawns. Thin wrapper around `agent-hardener setup`, which owns the garak version pin and venv
# location (single source of truth). garak is intentionally NOT an agent-hardener dependency (it
# pins litellm -> httpx>=0.28 + torch, which conflict with nvidia-nat), so it lives in its own
# venv. Idempotent; pass extra args (e.g. --force) through.
garak-venv *args:
    uv run agent-hardener setup {{ args }}

# Overlay a custom garak fork into the dedicated garak venv (shadows the PyPI release).
# Re-run `just garak-venv` after deleting ~/.agent-hardener/garak-venv to revert to the PyPI version.
[unix]
garak-dev path:
    uv pip install --python "$HOME/.agent-hardener/garak-venv/bin/python" -e {{ path }}

# Export dependencies to requirements.txt
export-reqs:
    @echo "Exporting dependencies to requirements.txt..."
    uv export --format requirements-txt --no-hashes --no-emit-project --output-file requirements.txt
    @echo "Requirements exported to requirements.txt"

# Start Phoenix and the OpenTelemetry Collector
[unix]
obs-start:
    ./scripts/start-observability.sh

# Start Phoenix and the OpenTelemetry Collector in the background
[unix]
obs-start-bg:
    ./scripts/start-observability.sh --background

# Stop Phoenix and the OpenTelemetry Collector
[unix]
obs-stop:
    ./scripts/stop-observability.sh

# Show observability stack status and trace file info
[unix]
obs-status:
    #!/usr/bin/env bash
    set -euo pipefail
    TRACE_FILE="agent_hardener/agents/victims/otellogs/llm_spans.json"
    echo "Observability Stack Status:"
    echo "==========================="
    if lsof -Pi :6006 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "Phoenix: running at http://localhost:6006"
    else
        echo "Phoenix: not running"
    fi
    if docker ps --format '{{ "{{" }}.Names{{ "}}" }}' 2>/dev/null | grep -q "^agent-hardener-otelcol$"; then
        echo "OTel: running in Docker container agent-hardener-otelcol"
    elif lsof -Pi :4318 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "OTel: port 4318 is listening"
    else
        echo "OTel: not running"
    fi
    if [ -f "$TRACE_FILE" ]; then
        echo "Trace file: $TRACE_FILE ($(wc -l < "$TRACE_FILE") lines)"
    else
        echo "Trace file: not created yet"
    fi

# Bootstrap and expose the OpenShell agents-lab victim on a Linux host
[unix]
openshell-bootstrap *args:
    ./scripts/bootstrap-openshell-victim.sh {{args}}

# Reset the OpenShell gateway from scratch and re-bootstrap the victim
[unix]
openshell-reset env_file=".env" *args:
    #!/usr/bin/env bash
    set -euo pipefail
    set -a
    . "{{env_file}}"
    set +a
    openshell gateway destroy --name auto-defender || true
    openshell gateway start --name auto-defender
    ./scripts/bootstrap-openshell-victim.sh --env-file "{{env_file}}" {{args}}


# Scaffold an agent-hardener.yaml manifest for your NAT agent
init *args:
    uv run agent-hardener init {{args}}

# Run Agent Hardener against your NAT agent (uses examples/agent-hardener.yaml by default)
[unix]
run *args:
    uv run agent-hardener run {{args}}

# Build and start the OpenShell sandbox for the configured agent
[unix]
up *args:
    uv run agent-hardener up {{args}}

# Tear down the OpenShell sandbox for the configured agent
[unix]
down *args:
    uv run agent-hardener down {{args}}

# Show OpenShell sandbox status for the configured agent
[unix]
status *args:
    uv run agent-hardener status {{args}}

# Run pre-commit hooks
hooks:
    uv run pre-commit run --all-files --hook-stage pre-commit --hook-stage pre-push

# Auto-fix lint issues
lint:
    @echo "Running ruff linter..."
    uv run ruff check --fix-only .

# Check-only linting (no fixes)
lint-check:
    @echo "Running ruff linter (check only)..."
    uv run ruff check --unsafe-fixes .

# Show lint statistics
lint-stats:
    @echo "Running ruff linter statistics..."
    uv run ruff check --statistics .

# Auto-format code
format:
    @echo "Running ruff formatter..."
    uv run ruff format .

# Check formatting without changes
format-check:
    @echo "Running ruff formatter (check only)..."
    uv run ruff format --diff .

# Scan for hardcoded secrets
secrets:
    uv run detect-secrets scan .

# Update third-party license information
update-licensecheck:
    uv run licensecheck --format=simple --file LICENSE-3rd-party.txt

# Run pytest tests
test:
    uv run pytest tests
    @echo "Testing completed."

# Run fast deterministic tests without slow/integration checks or coverage addopts
test-fast:
    uv run pytest tests -m "not slow and not integration" -q -o addopts=''

# Run the fast logical pipeline E2E
test-pipeline-e2e:
    uv run pytest tests/agent_hardener/test_pipeline_e2e.py -q -o addopts=''

# Run the optional real OpenShell smoke E2E
test-openshell-smoke-e2e:
    AGENT_HARDENER_RUN_OPEN_SHELL_SMOKE=1 uv run pytest tests/agent_hardener/test_openshell_smoke_e2e.py -q -o addopts=''

# Run pytest tests with JUnit XML report
test-junit:
    uv run pytest --junitxml=report.xml tests
    @echo "Testing completed."

# Run pytest tests with coverage report
test-coverage:
    uv run pytest \
        --cov \
        --cov-report=xml:coverage.xml \
        --cov-report term \
        --junitxml=report.xml \
        --disable-warnings \
        tests/
    @echo "Coverage reports generated (terminal + coverage.xml + report.xml)."

# Build wheel package
build:
    uv build
    @echo "Wheel package built in dist/ directory"

# NOTE: Recipes below use just's interpolation (double-brace syntax) and must
# stay inside raw/endraw blocks to prevent copier from interpreting them.
[unix]
[private]
bump type:
    #!/usr/bin/env bash
    set -euo pipefail
    OLD_VERSION=$(uv version | awk '{print $NF}')
    uv version --bump {{ type }} --no-sync
    NEW_VERSION=$(uv version | awk '{print $NF}')
    git add pyproject.toml uv.lock
    git commit -m "Bump version: $OLD_VERSION -> $NEW_VERSION"
    git tag "v$NEW_VERSION"
    echo "Bumped to $NEW_VERSION (tag: v$NEW_VERSION)"
    echo "Don't forget to push: git push && git push --tags"

[windows]
[private]
bump type:
    #!powershell
    $old = (uv version).Split()[-1]
    uv version --bump {{ type }} --no-sync
    $new = (uv version).Split()[-1]
    git add pyproject.toml uv.lock
    git commit -m "Bump version: $old -> $new"
    git tag "v$new"
    Write-Host "Bumped to $new (tag: v$new)"
    Write-Host "Don't forget to push: git push && git push --tags"

# Bump patch version, create commit and tag
bump-patch: (bump "patch")

# Bump minor version, create commit and tag
bump-minor: (bump "minor")

# Bump major version, create commit and tag
bump-major: (bump "major")

# Remove common cache files by glob-like patterns
[unix]
clean-globs:
    #!/usr/bin/env bash
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
    find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name ".DS_Store" -delete 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    find . -type f -name "*.pyo" -delete 2>/dev/null || true
    find . -type f -name "*.pyd" -delete 2>/dev/null || true

[windows]
clean-globs:
    #!powershell
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -Directory -Filter ".ipynb_checkpoints" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force
    Get-ChildItem -Recurse -File -Filter "*.pyc" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Recurse -File -Filter "*.pyo" -ErrorAction SilentlyContinue | Remove-Item -Force
    Get-ChildItem -Recurse -File -Filter "*.pyd" -ErrorAction SilentlyContinue | Remove-Item -Force

# Clean build artifacts, caches, and generated reports
clean: clean-globs
    {{ rm_rf }} .pytest_cache
    {{ rm_rf }} .coverage
    {{ rm_rf }} coverage.xml
    {{ rm_rf }} htmlcov/
    {{ rm_rf }} report.xml
    {{ rm_rf }} dist/
    {{ rm_rf }} build/
    {{ rm_rf }} .tox/

    @echo "Cleaned Python related files and cache"
