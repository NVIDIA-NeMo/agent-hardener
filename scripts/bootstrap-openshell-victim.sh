#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Bootstrap the OpenShell-backed victim described by an agent manifest on a fresh Linux host.
#
# This script is intentionally host-oriented: run it from the agent-hardener repo as
# the service user that should own the OpenShell gateway and sandbox.

set -euo pipefail

CONFIG="examples/agent-hardener.yaml"
# Empty means "the manifest's agent.secrets_file", which is the file the provider is created from.
ENV_FILE=""
# Empty means "take it from CONFIG" (resolved once the Python environment is synced).
GATEWAY=""
SANDBOX=""
PROVIDER=""
HEALTH_URL=""
EXTERNAL_HOST=""
RECREATE_PROVIDER=true
RECREATE_SANDBOX=true
START_OBSERVABILITY=false
SMOKE_CHAT=false
INSTALL_UV=true

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Bootstrap and expose the OpenShell victim described by an agent manifest. Only that manifest's
sandbox and provider are removed and recreated; other sandboxes on the gateway are left alone.

Options:
  --config PATH              Agent manifest or session config (default: $CONFIG)
  --env-file PATH            dotenv file with the manifest's secrets; must be the config's
                             agent.secrets_file (default: from the config)
  --gateway NAME             OpenShell gateway name (default: from the config)
  --sandbox NAME             OpenShell sandbox name (default: from the config)
  --provider NAME            OpenShell provider name (default: from the config)
  --health-url URL           Health URL to wait for (default: from the config)
  --external-host HOST       Host/IP to print in final curl commands
  --keep-provider            Do not delete/recreate the provider
  --keep-sandbox             Do not delete/recreate the sandbox before up
  --with-observability       Start Phoenix/OTel in the background first
  --smoke-chat               Run a chat-completions smoke test after health passes
  --no-install-uv            Fail if uv is missing instead of installing it
  -h, --help                 Show this help

Examples:
  ./scripts/bootstrap-openshell-victim.sh --config agent-hardener.yaml --external-host 10.220.128.11
  ./scripts/bootstrap-openshell-victim.sh --config agent-hardener.yaml --env-file /var/lib/openshellsvc/agent-hardener/.env --smoke-chat
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

log "Syncing Python environment"
uv sync --frozen

log "Reading the victim from $CONFIG"
resolved="$(uv run python - "$CONFIG" <<'PY'
import sys

from agent_hardener.config import load_config

cfg = load_config(sys.argv[1]).victim_control.config
print(cfg["gateway"])
print(cfg["sandbox"])
print(cfg["provider"])
print(cfg["health_url"])
print(" ".join(cfg.get("provider_credentials", [])))
print(cfg.get("provider_env_file") or "")
PY
)" || die "could not read $CONFIG"
{
  IFS= read -r cfg_gateway
  IFS= read -r cfg_sandbox
  IFS= read -r cfg_provider
  IFS= read -r cfg_health_url
  IFS= read -r cfg_credentials
  IFS= read -r cfg_env_file
} <<<"$resolved"
GATEWAY="${GATEWAY:-$cfg_gateway}"
SANDBOX="${SANDBOX:-$cfg_sandbox}"
PROVIDER="${PROVIDER:-$cfg_provider}"
HEALTH_URL="${HEALTH_URL:-$cfg_health_url}"
ENV_FILE="${ENV_FILE:-${cfg_env_file:-.env}}"
# `up` creates the provider from the config's secrets file, so validating any other file would check
# keys the victim never receives.
if [ -n "$cfg_env_file" ] && [ "$(realpath -m "$ENV_FILE")" != "$(realpath -m "$cfg_env_file")" ]; then
  die "--env-file $ENV_FILE differs from the config's agent.secrets_file ($cfg_env_file), which is what the provider is created from"
fi
[ -f "$ENV_FILE" ] || die "env file not found: $ENV_FILE"

log "Validating provider env file"
missing_keys=()
for key in $cfg_credentials; do
  if ! grep -Eq "^${key}=" "$ENV_FILE"; then
    missing_keys+=("$key")
  fi
done
if [ "${#missing_keys[@]}" -gt 0 ]; then
  die "missing required keys in $ENV_FILE: ${missing_keys[*]}"
fi
chmod 600 "$ENV_FILE" 2>/dev/null || true

if [ "$START_OBSERVABILITY" = true ]; then
  log "Starting observability stack"
  ./scripts/start-observability.sh --background
fi

log "Ensuring OpenShell gateway: $GATEWAY"
if ! openshell gateway info --gateway "$GATEWAY" >/dev/null 2>&1; then
  # The gateway is a system service; register it if needed.
  uv run agent-hardener setup || die "OpenShell gateway $GATEWAY is not reachable"
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
      sh -lc 'tail -200 /tmp/agent.log /tmp/relay-victim.log 2>/dev/null || true' >&2 || true
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

PORT="$(printf '%s' "$HEALTH_URL" | sed -E 's#^[a-z]+://[^:/]+:([0-9]+).*#\1#')"
printf '\nHealth:\n'
printf '  curl %s\n' "$HEALTH_URL"
if [ -n "$EXTERNAL_HOST" ]; then
  printf '  curl http://%s:%s/health\n' "$EXTERNAL_HOST" "$PORT"
  printf '\nChat Completions:\n'
  printf "  curl -sS -X POST http://%s:%s/v1/chat/completions -H 'Content-Type: application/json' --data '{\"model\":\"openshell-victim\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one short sentence.\"}]}'\n" "$EXTERNAL_HOST" "$PORT"
fi
