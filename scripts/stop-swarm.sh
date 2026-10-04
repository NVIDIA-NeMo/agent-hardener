#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#
# Stop the OpenShell/Garak Agent Hardener system while preserving the Docker daemon.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG="${AGENT_HARDENER_CONFIG:-$PROJECT_ROOT/examples/vulnerable_demo_e2e.yaml}"
STOP_GATEWAY=false

GATEWAY="${OPEN_SHELL_GATEWAY:-auto-defender}"
SANDBOX="${OPEN_SHELL_SANDBOX:-agents-lab-vulnerable-demo}"
FORWARD="${OPEN_SHELL_FORWARD:-0.0.0.0:8000}"
OPENSHELL_BIN="${OPEN_SHELL_BIN:-openshell}"

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Stop the Agent Hardener/OpenShell/Garak runtime system.

Options:
  --stop-gateway     Also terminate the local OpenShell gateway process.
                     By default the gateway is preserved because it is slow to load.
  -h, --help         Show this help.
EOF
}

log() {
  printf '\033[1;34m==>\033[0m %s\n' "$*"
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
    --stop-gateway)
      STOP_GATEWAY=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'error: unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

load_config_values() {
  [ -f "$CONFIG" ] || return 0

  local python_bin=""
  local candidate
  for candidate in "$PROJECT_ROOT/.venv/bin/python3" python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      python_bin="$candidate"
      break
    fi
  done
  [ -n "$python_bin" ] || die "config exists but no python interpreter was found to parse it: $CONFIG"

  local output=""
  local parse_error=""
  local parse_status=0
  parse_error="$(mktemp "${TMPDIR:-/tmp}/stop-swarm-config-parse.XXXXXX")"
  output="$("$python_bin" - "$CONFIG" 2>"$parse_error" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"failed to import PyYAML: {exc}")

try:
    config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
except Exception as exc:
    raise SystemExit(f"failed to parse config: {exc}")
victim_control = config.get("victim_control") if isinstance(config, dict) else {}
victim_config = victim_control.get("config") if isinstance(victim_control, dict) else {}
if not isinstance(victim_config, dict):
    victim_config = {}

for key in ("gateway", "sandbox", "forward", "openshell_bin"):
    value = victim_config.get(key) or ""
    print(f"{key}={value}")
PY
)" || parse_status=$?
  if [ "$parse_status" -ne 0 ]; then
    local detail
    detail="$(tr '\n' ' ' <"$parse_error" | sed 's/[[:space:]]*$//')"
    rm -f "$parse_error"
    die "failed to load config values from $CONFIG${detail:+: $detail}"
  fi
  rm -f "$parse_error"
  [ -n "$output" ] || return 0

  local key value
  while IFS='=' read -r key value; do
    case "$key" in
      gateway)
        [ -n "${OPEN_SHELL_GATEWAY:-}" ] || [ -z "$value" ] || GATEWAY="$value"
        ;;
      sandbox)
        [ -n "${OPEN_SHELL_SANDBOX:-}" ] || [ -z "$value" ] || SANDBOX="$value"
        ;;
      forward)
        [ -n "${OPEN_SHELL_FORWARD:-}" ] || [ -z "$value" ] || FORWARD="$value"
        ;;
      openshell_bin)
        [ -n "${OPEN_SHELL_BIN:-}" ] || [ -z "$value" ] || OPENSHELL_BIN="$value"
        ;;
    esac
  done <<<"$output"
}

process_command() {
  local pid="$1"
  ps -p "$pid" -o command= 2>/dev/null || true
}

is_numeric_pid() {
  case "$1" in
    ''|*[!0-9]*)
      return 1
      ;;
    *)
      return 0
      ;;
  esac
}

is_current_process() {
  local pid="$1"
  local current_bash_pid="${BASHPID:-$$}"
  [ "$pid" = "$$" ] || [ "$pid" = "$current_bash_pid" ] || [ "$pid" = "$PPID" ]
}

should_skip_command() {
  local command="$1"
  case "$command" in
    *"stop-swarm.sh"*)
      return 0
      ;;
    *"Docker.app"*|*"com.docker"*|*"dockerd"*|*"containerd"*)
      return 0
      ;;
  esac
  if [ "$STOP_GATEWAY" != true ]; then
    case "$command" in
      *"openshell-gateway"*)
        return 0
        ;;
    esac
  fi
  return 1
}

append_pid() {
  local pid="$1"
  local current="$2"
  case " $current " in
    *" $pid "*)
      printf '%s' "$current"
      ;;
    *)
      printf '%s %s' "$current" "$pid"
      ;;
  esac
}

terminate_pids() {
  local label="$1"
  local pids="$2"
  local use_group="${3:-false}"
  local live_pids=""
  local pid

  for pid in $pids; do
    is_numeric_pid "$pid" || continue
    is_current_process "$pid" && continue
    if kill -0 "$pid" 2>/dev/null; then
      live_pids="$(append_pid "$pid" "$live_pids")"
    fi
  done

  if [ -z "$live_pids" ]; then
    return 0
  fi

  log "Stopping $label:$live_pids"
  for pid in $live_pids; do
    if [ "$use_group" = true ]; then
      kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    else
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done

  local attempt
  for attempt in 1 2 3 4 5; do
    local any_alive=false
    for pid in $live_pids; do
      if kill -0 "$pid" 2>/dev/null; then
        any_alive=true
        break
      fi
    done
    [ "$any_alive" = true ] || return 0
    sleep 1
  done

  for pid in $live_pids; do
    if kill -0 "$pid" 2>/dev/null; then
      warn "Force killing stubborn $label process: $pid"
      if [ "$use_group" = true ]; then
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      else
        kill -KILL "$pid" 2>/dev/null || true
      fi
    fi
  done
}

stop_pid_file() {
  local pid_file="$1"
  local label="$2"
  local expected="$3"
  local use_group="${4:-false}"

  [ -f "$pid_file" ] || return 0

  local raw_pid
  raw_pid="$(tr -d '[:space:]' <"$pid_file" 2>/dev/null || true)"
  if ! is_numeric_pid "$raw_pid"; then
    warn "Removing invalid PID file: $pid_file"
    rm -f "$pid_file"
    return 0
  fi

  local command
  command="$(process_command "$raw_pid")"
  if [ -z "$command" ]; then
    rm -f "$pid_file"
    return 0
  fi
  case "$command" in
    *"$expected"*)
      if should_skip_command "$command"; then
        return 0
      fi
      terminate_pids "$label" "$raw_pid" "$use_group"
      ;;
    *)
      warn "Removing stale PID file with reused PID: $pid_file"
      ;;
  esac
  rm -f "$pid_file"
}

kill_matching() {
  local label="$1"
  shift

  local pids=""
  local pid command needle matched
  while read -r pid command; do
    is_numeric_pid "$pid" || continue
    is_current_process "$pid" && continue
    [ -n "$command" ] || continue
    should_skip_command "$command" && continue

    matched=true
    for needle in "$@"; do
      case "$command" in
        *"$needle"*) ;;
        *)
          matched=false
          break
          ;;
      esac
    done
    if [ "$matched" = true ]; then
      pids="$(append_pid "$pid" "$pids")"
    fi
  done < <(ps -axo pid=,command= 2>/dev/null || true)

  terminate_pids "$label" "$pids"
}

kill_port_listeners() {
  local port="$1"
  if ! have lsof; then
    warn "lsof not found; cannot check port $port"
    return 0
  fi

  local pids=""
  local pid command
  while IFS= read -r pid; do
    is_numeric_pid "$pid" || continue
    is_current_process "$pid" && continue
    command="$(process_command "$pid")"
    [ -n "$command" ] || continue
    should_skip_command "$command" && continue
    pids="$(append_pid "$pid" "$pids")"
  done < <(lsof -nP -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)

  terminate_pids "listeners on port $port" "$pids"
}

is_openshell_gateway_container() {
  local metadata="$1"
  case "$metadata" in
    *"openshell-gateway"*|*"openshell_gateway"*|*"openshell/gateway"*|*"openshell gateway"*|*"gateway openshell"*)
      return 0
      ;;
  esac
  return 1
}

stop_openshell_resources() {
  if ! have "$OPENSHELL_BIN"; then
    warn "OpenShell CLI not found: $OPENSHELL_BIN"
    return 0
  fi

  if [ -n "$FORWARD" ]; then
    log "Stopping OpenShell forward $FORWARD on gateway $GATEWAY"
    "$OPENSHELL_BIN" forward stop "$FORWARD" --gateway "$GATEWAY" >/dev/null 2>&1 || true
  fi

  if [ -n "$SANDBOX" ]; then
    log "Deleting OpenShell sandbox $SANDBOX on gateway $GATEWAY"
    "$OPENSHELL_BIN" sandbox delete "$SANDBOX" --gateway "$GATEWAY" >/dev/null 2>&1 || true
  fi
}

stop_all_docker_containers() {
  if ! have docker; then
    warn "Docker CLI not found; skipping Docker container stop"
    return 0
  fi

  local docker_error
  local docker_status=0
  local containers
  docker_error="$(mktemp "${TMPDIR:-/tmp}/stop-swarm-docker.XXXXXX")"
  containers="$(docker ps --format '{{.ID}}\t{{.Names}}\t{{.Image}}\t{{.Command}}\t{{.Labels}}' 2>"$docker_error")" \
    || docker_status=$?
  if [ "$docker_status" -ne 0 ]; then
    local detail
    detail="$(tr '\n' ' ' <"$docker_error" | sed 's/[[:space:]]*$//')"
    rm -f "$docker_error"
    warn "docker ps failed; Docker containers were not stopped${detail:+: $detail}"
    return 0
  fi
  rm -f "$docker_error"
  if [ -z "$containers" ]; then
    log "No running Docker containers found"
    return 0
  fi

  local container_ids=""
  local preserved_gateway_ids=""
  local id name image command labels metadata
  while IFS=$'\t' read -r id name image command labels; do
    [ -n "$id" ] || continue
    metadata="$name $image $command $labels"
    if [ "$STOP_GATEWAY" != true ] && is_openshell_gateway_container "$metadata"; then
      preserved_gateway_ids="${preserved_gateway_ids:+$preserved_gateway_ids }$id"
      continue
    fi
    container_ids="${container_ids:+$container_ids }$id"
  done <<<"$containers"

  if [ -n "$preserved_gateway_ids" ]; then
    log "Preserving OpenShell gateway container(s): $preserved_gateway_ids"
  fi
  if [ -z "$container_ids" ]; then
    log "No non-gateway Docker containers to stop"
    return 0
  fi

  log "Stopping all running Docker containers"
  # Intentionally unquoted: docker stop expects one argument per container id.
  docker stop $container_ids >/dev/null 2>&1 || warn "docker stop failed; Docker may be unavailable"
}

load_config_values
cd "$PROJECT_ROOT"

log "Stopping Agent Hardener runtime"
printf 'config: %s\n' "$CONFIG"
printf 'gateway: %s%s\n' "$GATEWAY" "$([ "$STOP_GATEWAY" = true ] && printf ' (will stop)' || printf ' (preserved)')"
printf 'sandbox: %s\n' "$SANDBOX"

shopt -s nullglob
for pid_file in "$PROJECT_ROOT"/.agent-hardener/openshell-logs/*-forward-*.pid; do
  stop_pid_file "$pid_file" "managed OpenShell forward" "openshell forward start" true
done
for pid_file in "$PROJECT_ROOT"/.agent-hardener/run-logs/*/garak-api.pid; do
  stop_pid_file "$pid_file" "Garak API" "uvicorn app:app"
done
for pid_file in "$PROJECT_ROOT"/agent_hardener/agents/victims/otellogs/phoenix.pid; do
  stop_pid_file "$pid_file" "Phoenix" "phoenix serve"
done
shopt -u nullglob

kill_matching "Agent Hardener wrapper" "run-openshell-garak-cycle.sh"
kill_matching "Agent Hardener OpenShell runs" "agent_hardener.tools.openshell" "run"
kill_matching "Agent Hardener run-summary filters" "agent_hardener.tools.run_summary"
kill_matching "Garak API" "uvicorn app:app" "--port 8010"
kill_matching "Garak scans" "-m garak" "--config scan_"
kill_matching "OpenShell forwards" "openshell forward start"
kill_matching "OpenShell SSH proxies" "openshell ssh-proxy"
kill_matching "Phoenix" "phoenix serve"

stop_openshell_resources
stop_all_docker_containers
rm -f "$PROJECT_ROOT"/agent_hardener/agents/victims/otellogs/otelcol.container

for port in 8000 8010 4318 6006; do
  kill_port_listeners "$port"
done

if [ "$STOP_GATEWAY" = true ]; then
  kill_matching "OpenShell gateway" "openshell-gateway"
else
  log "OpenShell gateway preserved"
fi

log "Swarm stop complete"
