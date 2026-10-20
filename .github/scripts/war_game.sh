#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run one full war-game through NeMo Helix (`nemo agent-hardener init -> synth-benign -> run`)
# against one of Helix's example agents, and leave <agent>.validation.json in the output dir.
#
# Usage: war_game.sh <langgraph|relay> <nemo-helix checkout> <output dir>
# Needs: INFERENCE_API_KEY (a build.nvidia.com key), Docker, the OpenShell CLI, Helix bootstrapped.

set -euo pipefail

AGENT=$1
HELIX_DIR=$(cd "$2" && pwd)
OUT_DIR=$(mkdir -p "$3" && cd "$3" && pwd)
: "${INFERENCE_API_KEY:?INFERENCE_API_KEY must be set}"

# Inference Hub is internal-only, so victims are pointed at the public build.nvidia.com endpoint.
INFERENCE_HOST=integrate.api.nvidia.com
INFERENCE_BASE_URL=https://${INFERENCE_HOST}/v1
VICTIM_MODEL=${VICTIM_MODEL:-nvidia/nemotron-3.5-lightning-30b-a3b}
# agent-hardener's default (nemotron-3-super-120b) is listed on build.nvidia.com but 404s on chat
# for this key, so both roles are pinned; the guardrail judge falls back to the analysis model.
MODEL_FLAGS=(
    --attack-model "${ATTACK_MODEL:-nvidia/nemotron-3-ultra-550b-a55b}"
    --analysis-model "${ANALYSIS_MODEL:-$VICTIM_MODEL}"
)
EXAMPLES_DIR=$HELIX_DIR/plugins/nemo-agent-hardener/examples
WORK_DIR=$OUT_DIR/work
ATTACK_INTENSITY=${ATTACK_INTENSITY:-light}

# Everything shares one build.nvidia.com key, which allows ~100 requests/min but only a burst of
# ~30, so defenders and validators work one item at a time instead of fanning out.
export AGENT_HARDENER_DEFENDER_CONCURRENCY=${AGENT_HARDENER_DEFENDER_CONCURRENCY:-1}
export AGENT_HARDENER_VALIDATOR_CONCURRENCY=${AGENT_HARDENER_VALIDATOR_CONCURRENCY:-1}
export AGENT_HARDENER_ATTACK_VALIDATOR_CONCURRENCY=${AGENT_HARDENER_ATTACK_VALIDATOR_CONCURRENCY:-1}
export AGENT_HARDENER_BENIGN_VALIDATOR_CONCURRENCY=${AGENT_HARDENER_BENIGN_VALIDATOR_CONCURRENCY:-1}

export NHX_BASE_URL=http://localhost:8080
export PATH=$HELIX_DIR/.venv/bin:$PATH

log() { printf '\n==> %s\n' "$*"; }

start_platform() {
    log "Starting NeMo Helix"
    # 0.0.0.0 so the OpenShell sandbox can reach the gateway through host.docker.internal.
    (cd "$HELIX_DIR" && nohup nemo services run --service-group all --controllers models,jobs \
        --host 0.0.0.0 --port 8080 >"$OUT_DIR/platform.log" 2>&1 &)
    for _ in $(seq 1 150); do
        curl -sf "$NHX_BASE_URL/health/ready" >/dev/null && return 0
        sleep 2
    done
    echo "Platform did not become ready; see platform.log" >&2
    return 1
}

register_provider() {
    log "Registering the build.nvidia.com provider"
    printf '%s' "$INFERENCE_API_KEY" | nemo secrets create nvidia-inference-key --from-file - --workspace default
    nemo inference providers create nvidia-inference --workspace default \
        --host-url "$INFERENCE_BASE_URL" --api-key-secret-name nvidia-inference-key
}

setup_agent_hardener() {
    log "Installing agent-hardener from ${NEMO_AGENT_HARDENER_SPEC:-PyPI}"
    nemo agent-hardener setup --force
    nemo agent-hardener doctor
}

init_langgraph() {
    MANIFEST=langgraph-victim
    nemo agent-hardener init --project-dir "$EXAMPLES_DIR/langgraph-victim" --name "$MANIFEST" \
        --harness langgraph --relay-confirmed --yes \
        --start-command "/usr/local/bin/python /app/server.py" --binary "/usr/local/bin/python*" \
        --secrets INFERENCE_API_KEY --egress "$INFERENCE_HOST" \
        --env "INFERENCE_BASE_URL=$INFERENCE_BASE_URL" --env "INFERENCE_MODEL=$VICTIM_MODEL" \
        "${MODEL_FLAGS[@]}"
}

init_relay() {
    MANIFEST=ledger
    # agent.yaml hard-codes Inference Hub, so register a patched copy; the Helix checkout stays untouched.
    local agent_dir=$WORK_DIR/relay-victim
    cp -R "$EXAMPLES_DIR/relay-victim" "$agent_dir"
    # `-i.orig` is the in-place form GNU and BSD sed both accept.
    sed -i.orig \
        -e "s|base_url: https://inference-api.nvidia.com/v1|base_url: $INFERENCE_BASE_URL|" \
        -e "s|model: nvidia/nvidia/nemotron-3.5-lightning-30b-a3b|model: $VICTIM_MODEL|" \
        "$agent_dir/agent.yaml"
    grep -q "$INFERENCE_BASE_URL" "$agent_dir/agent.yaml" || { echo "agent.yaml patch did not apply" >&2; return 1; }

    nemo agents package --agent "$agent_dir/agent.yaml" --dockerfile "$agent_dir/Dockerfile" --tag ledger:v1
    nemo agents create --name ledger --agent-config "$agent_dir/agent.yaml"
    # mcp.deepwiki.com is the agent's second, remote MCP server.
    nemo agent-hardener init --agent ledger --name "$MANIFEST" --yes \
        --secrets INFERENCE_API_KEY --egress "$INFERENCE_HOST" --egress mcp.deepwiki.com \
        "${MODEL_FLAGS[@]}"
}

# The war-game runs in a job temp dir; run.json's log artifact points into it, and the run's
# .agent-hardener/run-logs/<run_id>/ sits beside the results directory.
run_logs_dir() {
    python3 -c '
import json, sys
from pathlib import Path
from urllib.parse import urlparse
url = json.load(open(sys.argv[1]))["results"]["agent-hardener-log"]["artifact_url"]
print(Path(urlparse(url).path).parents[1] / ".agent-hardener" / "run-logs")
' "$OUT_DIR/run.json"
}

run_error_category() {
    python3 -c 'import json, sys; print((json.load(open(sys.argv[1])).get("error") or {}).get("category", ""))' \
        "$OUT_DIR/run.json"
}

collect_results() {
    local run_logs validation
    run_logs=$(run_logs_dir) || { echo "run.json has no log artifact" >&2; return 1; }
    cp -R "$run_logs" "$OUT_DIR/run-logs"
    # Newest first; `ls -t` rather than `find -printf`, which BSD find lacks.
    validation=$(find "$OUT_DIR/run-logs" -name validation.json -exec ls -t {} + | head -1)
    if [[ -z $validation ]]; then
        echo "No validation.json under $run_logs" >&2
        return 1
    fi
    cp "$validation" "$OUT_DIR/$AGENT.validation.json"
}

main() {
    mkdir -p "$WORK_DIR"
    cd "$WORK_DIR"
    start_platform
    register_provider
    setup_agent_hardener

    log "Initialising the $AGENT victim"
    case $AGENT in
        langgraph) init_langgraph ;;
        relay) init_relay ;;
        *) echo "Unknown agent: $AGENT" >&2; exit 2 ;;
    esac

    log "Synthesising the benign suite"
    nemo agent-hardener synth-benign --manifest-id "$MANIFEST" --no-interactive

    log "Running the war-game"
    # A failed run still leaves partial logs worth collecting, so the status is checked after.
    local status=0
    nemo agent-hardener run --manifest-id "$MANIFEST" --attack-intensity "$ATTACK_INTENSITY" >"$OUT_DIR/run.json" || status=$?
    collect_results
    # `run` fails a round unless every attack is blocked and every benign request passes; the
    # thresholds are check_war_game.py's call, so only other failures fail this script.
    if [[ $status -ne 0 && $(run_error_category) == validation_failed ]]; then
        status=0
    fi
    return "$status"
}

main
