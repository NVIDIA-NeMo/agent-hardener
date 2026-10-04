#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Bootstrap the OpenShell-backed agents-lab victim on a fresh Linux host.
#
# This script is intentionally host-oriented: run it from the agent-hardener repo as
# the service user that should own the OpenShell gateway and sandbox.

set -euo pipefail

CONFIG="examples/vulnerable_demo_e2e.yaml"
ENV_FILE=".env"
GATEWAY="auto-defender"
SANDBOX="agents-lab-vulnerable-demo"
PROVIDER="agents-lab-secrets"
HEALTH_URL="http://127.0.0.1:8000/health"
EXTERNAL_HOST=""
RECREATE_PROVIDER=true
RECREATE_SANDBOX=true
START_OBSERVABILITY=false
SMOKE_CHAT=false
INSTALL_UV=true

REQUIRED_ENV_KEYS=(
  NIM_API_KEY
  INFERENCE_API_KEY
  VLLM_API_KEY
  GITHUB_TOKEN
  GITHUB_REPOSITORY
)

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap and expose the OpenShell agents-lab victim.

Options:
  --config PATH              Agent Hardener config (default: $CONFIG)
  --env-file PATH            dotenv file with provider credentials (default: $ENV_FILE)
  --gateway NAME             OpenShell gateway name (default: $GATEWAY)
  --sandbox NAME             OpenShell sandbox name (default: $SANDBOX)
  --provider NAME            OpenShell provider name (default: $PROVIDER)
  --health-url URL           Health URL to wait for (default: $HEALTH_URL)
  --external-host HOST       Host/IP to print in final curl commands
  --keep-provider            Do not delete/recreate the provider
  --keep-sandbox             Do not delete/recreate the sandbox before up
  --with-observability       Start Phoenix/OTel in the background first
  --smoke-chat               Run a chat-completions smoke test after health passes
  --no-install-uv            Fail if uv is missing instead of installing it
  -h, --help                 Show this help

Examples:
  ./scripts/bootstrap-openshell-victim.sh --external-host 10.220.128.11
  ./scripts/bootstrap-openshell-victim.sh --env-file /var/lib/openshellsvc/agent-hardener/.env --smoke-chat
EOF
}

log() {
  printf '\n\033[1;34m==>\033[0m %s\n' "$*"
}

warn() {
  printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2
}

die() {
  printf '\033[1;31merror:\033[0m %s\n' "$*" >&2
  exit 1
}

have() {
  command -v "$1" >/dev/null 2>&1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --gateway)
      GATEWAY="$2"
      shift 2
      ;;
    --sandbox)
      SANDBOX="$2"
      shift 2
      ;;
    --provider)
      PROVIDER="$2"
      shift 2
      ;;
    --health-url)
      HEALTH_URL="$2"
      shift 2
      ;;
    --external-host)
      EXTERNAL_HOST="$2"
      shift 2
      ;;
    --keep-provider)
      RECREATE_PROVIDER=false
      shift
      ;;
    --keep-sandbox)
      RECREATE_SANDBOX=false
      shift
      ;;
    --with-observability)
      START_OBSERVABILITY=true
      shift
      ;;
    --smoke-chat)
      SMOKE_CHAT=true
      shift
      ;;
    --no-install-uv)
      INSTALL_UV=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown option: $1"
      ;;
  esac
done

[ -f "$CONFIG" ] || die "config not found: $CONFIG"
[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"

if [ -z "$EXTERNAL_HOST" ] && have hostname; then
  EXTERNAL_HOST="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
fi

log "Checking local prerequisites"
have curl || die "curl is required"
have git || warn "git not found; continuing because the repo is already present"
have docker || die "docker is required"
docker info >/dev/null 2>&1 || die "docker is not reachable for user $(id -un)"

if ! have uv; then
  if [ "$INSTALL_UV" = false ]; then
    die "uv is required"
  fi
  log "Installing uv for $(id -un)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
have uv || die "uv installation did not put uv on PATH"

have openshell || die "openshell CLI is required on PATH"

log "Validating provider env file"
missing_keys=()
for key in "${REQUIRED_ENV_KEYS[@]}"; do
  if ! grep -Eq "^${key}=" "$ENV_FILE"; then
    missing_keys+=("$key")
  fi
done
if [ "${#missing_keys[@]}" -gt 0 ]; then
  die "missing required keys in $ENV_FILE: ${missing_keys[*]}"
fi
chmod 600 "$ENV_FILE" 2>/dev/null || true

log "Syncing Python environment"
uv sync --frozen

if [ "$START_OBSERVABILITY" = true ]; then
  log "Starting observability stack"
  ./scripts/start-observability.sh --background
fi

log "Ensuring OpenShell gateway: $GATEWAY"
if ! openshell gateway info --gateway "$GATEWAY" >/dev/null 2>&1; then
  openshell gateway start --name "$GATEWAY"
fi

if [ "$RECREATE_SANDBOX" = true ]; then
  log "Removing existing sandbox if present: $SANDBOX"
  openshell sandbox delete "$SANDBOX" --gateway "$GATEWAY" >/dev/null 2>&1 || true
fi

if [ "$RECREATE_PROVIDER" = true ]; then
  log "Refreshing provider credentials: $PROVIDER"
  openshell provider delete "$PROVIDER" --gateway "$GATEWAY" >/dev/null 2>&1 || true
fi

log "Creating/updating OpenShell victim"
uv run python -m agent_hardener.tools.openshell up --config "$CONFIG"

log "Waiting for victim health: $HEALTH_URL"
deadline=$((SECONDS + 120))
until curl -fsS "$HEALTH_URL" >/dev/null 2>&1; do
  if [ "$SECONDS" -ge "$deadline" ]; then
    openshell sandbox exec --gateway "$GATEWAY" --name "$SANDBOX" --no-tty -- \
      sh -lc 'tail -200 /tmp/agents-lab.log 2>/dev/null || true' >&2 || true
    die "timed out waiting for $HEALTH_URL"
  fi
  sleep 2
done

if [ "$SMOKE_CHAT" = true ]; then
  log "Running chat completions smoke test"
  curl -fsS -m 120 \
    -X POST "${HEALTH_URL%/health}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    --data '{"model":"openshell-victim","messages":[{"role":"user","content":"Say hello in one short sentence."}]}'
  printf '\n'
fi

log "OpenShell victim is ready"
openshell forward list --gateway "$GATEWAY"

printf '\nHealth:\n'
printf '  curl %s\n' "$HEALTH_URL"
if [ -n "$EXTERNAL_HOST" ]; then
  printf '  curl http://%s:8000/health\n' "$EXTERNAL_HOST"
  printf '\nChat Completions:\n'
  printf "  curl -sS -X POST http://%s:8000/v1/chat/completions -H 'Content-Type: application/json' --data '{\"model\":\"openshell-victim\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}]}'\n" "$EXTERNAL_HOST"
fi
