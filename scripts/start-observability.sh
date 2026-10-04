#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/agent_hardener/agents/victims/otellogs"
OTEL_CONFIG="$PROJECT_ROOT/agent_hardener/agents/victims/otel-collector-config.yaml"
PHOENIX_PORT=6006
OTEL_HTTP_PORT=4318
CONTAINER_NAME=agent-hardener-otelcol
BACKGROUND=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b|--background)
            BACKGROUND=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-b|--background]"
            exit 1
            ;;
    esac
done

mkdir -p "$LOGS_DIR"

echo "Starting observability stack"
echo "============================"

if lsof -Pi :$PHOENIX_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "Phoenix already running on port $PHOENIX_PORT"
else
    echo "Starting Phoenix server on port $PHOENIX_PORT..."
    export PHOENIX_GRPC_PORT=0
    if [ "$BACKGROUND" = true ]; then
        nohup uv run phoenix serve > "$LOGS_DIR/phoenix.log" 2>&1 &
        echo $! > "$LOGS_DIR/phoenix.pid"
        echo "Logs: $LOGS_DIR/phoenix.log"
    else
        uv run phoenix serve &
        echo $! > "$LOGS_DIR/phoenix.pid"
    fi
    sleep 3
fi

if lsof -Pi :$OTEL_HTTP_PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "OpenTelemetry Collector already running on port $OTEL_HTTP_PORT"
else
    echo "Starting OpenTelemetry Collector with Docker..."

    if ! command -v docker >/dev/null 2>&1; then
        echo "Docker not found. Please install Docker to run the OpenTelemetry Collector."
        if [ -f "$LOGS_DIR/phoenix.pid" ]; then
            kill "$(cat "$LOGS_DIR/phoenix.pid")" 2>/dev/null || true
        fi
        exit 1
    fi

    if ! docker image inspect otel/opentelemetry-collector:latest >/dev/null 2>&1; then
        echo "Pulling OpenTelemetry Collector Docker image..."
        docker pull otel/opentelemetry-collector:latest
    fi

    CONTAINER_ID=$(docker run -d \
        --name "$CONTAINER_NAME" \
        --rm \
        -p 127.0.0.1:4317:4317 \
        -p 127.0.0.1:4318:4318 \
        -v "$OTEL_CONFIG:/etc/otel-collector-config.yaml:ro" \
        -v "$LOGS_DIR:/otellogs" \
        otel/opentelemetry-collector:latest \
        --config /etc/otel-collector-config.yaml)

    echo "$CONTAINER_ID" > "$LOGS_DIR/otelcol.container"
    echo "Collector container: ${CONTAINER_ID:0:12}"
    sleep 2
fi

echo ""
echo "Observability stack is ready"
echo "Phoenix UI:    http://localhost:$PHOENIX_PORT"
echo "Trace file:    $LOGS_DIR/llm_spans.json"
echo "OTel endpoint: http://localhost:$OTEL_HTTP_PORT"
echo ""

if [ "$BACKGROUND" = false ]; then
    echo "Press Ctrl+C to stop all foreground services"
    wait
fi
