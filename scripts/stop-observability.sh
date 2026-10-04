#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/agent_hardener/agents/victims/otellogs"
CONTAINER_NAME=agent-hardener-otelcol

echo "Stopping observability stack"
echo "============================"

if [ -f "$LOGS_DIR/phoenix.pid" ]; then
    PID="$(cat "$LOGS_DIR/phoenix.pid")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Phoenix (PID: $PID)..."
        kill "$PID"
    fi
    rm -f "$LOGS_DIR/phoenix.pid"
else
    echo "Phoenix PID not found; leaving any external Phoenix process alone"
fi

if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping OpenTelemetry Collector (Docker)..."
    docker stop "$CONTAINER_NAME" >/dev/null
    rm -f "$LOGS_DIR/otelcol.container"
else
    echo "OpenTelemetry Collector not running"
fi

echo "All services stopped"
