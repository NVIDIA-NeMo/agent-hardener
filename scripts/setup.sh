#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# First-time setup: install dependencies, configure OpenShell, and create .env files.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OS="$(uname -s)"

cd "$PROJECT_ROOT"

green='\033[1;32m'; blue='\033[1;34m'; yellow='\033[1;33m'; red='\033[1;31m'; reset='\033[0m'
log()  { printf "\n${blue}==>%b %s\n" "${reset}" "$*"; }
ok()   { printf "${green}✓%b %s\n" "${reset}" "$*"; }
warn() { printf "${yellow}warning:%b %s\n" "${reset}" "$*"; }
die()  { printf "${red}error:%b %s\n" "${reset}" "$*" >&2; exit 1; }

ask() {
    local var="$1" prompt="$2" hint="$3" secret="${4:-false}"
    while true; do
        printf "\n  ${blue}%s${reset}\n  ${yellow}hint: %s${reset}\n  > " "$prompt" "$hint"
        if [ "$secret" = "true" ]; then  # pragma: allowlist secret
            read -r -s value; echo
        else
            read -r value
        fi
        [ -n "$value" ] && break
        warn "value cannot be empty, please try again"
    done
    eval "$var=\"\$value\""
}

# ── Prerequisites ─────────────────────────────────────────────────────────────
log "Checking prerequisites"
command -v curl >/dev/null 2>&1   || die "curl is required. Install with: sudo apt-get install -y curl"
command -v docker >/dev/null 2>&1 || die "Docker is required. See https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1       || die "Docker daemon is not running. Start Docker first."
if [ "$OS" = "Darwin" ]; then
    command -v brew >/dev/null 2>&1 || die "Homebrew is required. Install from https://brew.sh"
fi
command -v openshell >/dev/null 2>&1 || die "OpenShell CLI is required. Run: curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh"
_openshell_link=$(readlink "$(command -v openshell)" 2>/dev/null || true)
case "$(command -v openshell) ${_openshell_link}" in
    *uv/tools*) die "openshell is installed via 'uv tool install', which only provides the CLI without the gateway service. Remove it with: uv tool uninstall openshell   Then install the native package: curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh" ;;
esac
ok "Prerequisites present"

# ── Stop any previous run ─────────────────────────────────────────────────────
log "Stopping any previous Agent Hardener run"
"$SCRIPT_DIR/stop-swarm.sh" 2>/dev/null || true

# ── Victim port selection ─────────────────────────────────────────────────────
_port_in_use() {
    lsof -iTCP:"$1" -sTCP:LISTEN -t >/dev/null 2>&1 || \
    ss -tlnp 2>/dev/null | grep -q ":$1[[:space:]]"
}
VICTIM_PORT=8000
if _port_in_use "$VICTIM_PORT"; then
    _i=0
    printf "  Waiting for port %s to be released" "$VICTIM_PORT"
    while _port_in_use "$VICTIM_PORT" && [ $_i -lt 10 ]; do
        printf "."
        sleep 1
        _i=$((_i + 1))
    done
    printf "\n"
fi
if _port_in_use "$VICTIM_PORT"; then
    warn "Port ${VICTIM_PORT} is already in use."
    while true; do
        printf "  Enter a free port for the victim agent (e.g. 8001): "
        read -r VICTIM_PORT
        case "$VICTIM_PORT" in
            ''|*[!0-9]*) warn "Invalid port — enter a number between 1024 and 65535"; continue ;;
        esac
        [ "$VICTIM_PORT" -ge 1024 ] && [ "$VICTIM_PORT" -le 65535 ] || { warn "Port out of range"; continue; }
        _port_in_use "$VICTIM_PORT" && { warn "Port ${VICTIM_PORT} is also in use, try another"; continue; }
        break
    done
    ok "Using port ${VICTIM_PORT} for victim agent"
fi

# ── uv ────────────────────────────────────────────────────────────────────────
if ! command -v uv >/dev/null 2>&1; then
    log "Installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv install succeeded but uv is not on PATH — open a new shell or add ~/.local/bin to PATH, then re-run"
    ok "uv installed"
fi

# OpenShell gateway registration happens in `agent-hardener setup`, run by `just install` below.

# ── Root .env ─────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    [ -w "$PROJECT_ROOT" ] || die "Cannot write to $PROJECT_ROOT — run: chmod u+w \"$PROJECT_ROOT\""
    log "Creating .env — please provide your credentials"

    ask INFERENCE_API_KEY \
        "INFERENCE_API_KEY — NIM inference API key (used by the victim NAT workflow and defenders)" \
        "starts with sk-  e.g. sk-abc123..."

    ask GITHUB_REPOSITORY \
        "GITHUB_REPOSITORY — GitHub repo the research agent will read issues from" \
        "format: owner/repo  e.g. myorg/my-test-repo"

    ask GITHUB_TOKEN \
        "GITHUB_TOKEN — GitHub personal access token (repo read + issue write scope)" \
        "starts with github_pat_  or  ghp_  e.g. github_pat_11ABC..."

    cat > .env <<EOF
# Required for the research victim NAT workflow.
INFERENCE_API_KEY="${INFERENCE_API_KEY}"
GITHUB_REPOSITORY=${GITHUB_REPOSITORY}
GITHUB_TOKEN="${GITHUB_TOKEN}"

# Optional keys used by OpenShell and defender examples.
NIM_API_KEY="${INFERENCE_API_KEY}"
VLLM_API_KEY=dummy
EOF
    ok "Created .env"
else
    ok ".env already exists, skipping"
fi

# ── Install dependencies ──────────────────────────────────────────────────────
log "Installing dependencies"
just install

DEMO_CONFIG="examples/vulnerable_demo_e2e.yaml"
if [ "$VICTIM_PORT" != "8000" ]; then
    mkdir -p .agent-hardener
    DEMO_CONFIG=".agent-hardener/demo-config.yaml"
    grep -q ":8000" examples/vulnerable_demo_e2e.yaml \
        || warn "':8000' not found in examples/vulnerable_demo_e2e.yaml — the port rewrite may be incomplete"
    sed \
        -e "s/:8000/:${VICTIM_PORT}/g" \
        -e "s|start_command: /app/start-agents-lab.sh|start_command: env VICTIM_PORT=${VICTIM_PORT} /app/start-agents-lab.sh|" \
        examples/vulnerable_demo_e2e.yaml > "$DEMO_CONFIG"
    ok "Config rewritten with port ${VICTIM_PORT} at ${DEMO_CONFIG}"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
ok "Setup complete — next steps:"
printf "\n  uv run agent-hardener init           # scaffold a manifest for your agent\n"
printf "  uv run agent-hardener synth-benign   # generate the benign test suite\n"
printf "  uv run agent-hardener run            # run Agent Hardener against your agent\n\n"
