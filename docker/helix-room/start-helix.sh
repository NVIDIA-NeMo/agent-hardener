#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Room start hook: `nemo services start` runs the NeMo Helix services in the background on :8080
# (AH_ROOM_HELIX_PORT) and waits until they are ready. NHX_ROOM_HOST=0.0.0.0 makes a published
# -p 127.0.0.1:8080:8080 reach them; the default keeps them on the room's loopback. With AH_ROOM_DOCKER=host the
# room shares the host's network, so the port is the host's: it must be free, and NHX_BASE_URL must
# name it.
set -euo pipefail

HOST="${NHX_ROOM_HOST:-127.0.0.1}"
PORT="${AH_ROOM_HELIX_PORT:-8080}"
TIMEOUT="${AH_ROOM_HELIX_TIMEOUT:-300}"

die() {
  printf 'ah-room: %s\n' "$*" >&2
  exit 1
}

[[ "${HOST}" == 127.0.0.1 || "${HOST}" == 0.0.0.0 ]] || die "NHX_ROOM_HOST must be 127.0.0.1 or 0.0.0.0"
# On the host's network 0.0.0.0 would open Studio, which has no login, to every machine that reaches this one.
[[ "${AH_ROOM_DOCKER:-private}" != host || "${HOST}" == 127.0.0.1 ]] \
  || die "with AH_ROOM_DOCKER=host Studio listens on 127.0.0.1 only; reach it from elsewhere with ssh -L"
[[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT >= 1 && PORT <= 65535 )) || die "AH_ROOM_HELIX_PORT must be 1-65535"
[[ "${NHX_BASE_URL:-}" == "http://127.0.0.1:${PORT}" ]] \
  || die "NHX_BASE_URL is '${NHX_BASE_URL:-}' but Helix runs on port ${PORT}; set NHX_BASE_URL=http://127.0.0.1:${PORT}"
if [[ "${AH_ROOM_DOCKER:-private}" == host ]] \
  && [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:${PORT}/")" != "000" ]]; then
  die "port ${PORT} is taken on this host; pick another (ah-room up --port, or AH_ROOM_HELIX_PORT)"
fi

cd /opt/nemo-helix
nemo services start --instance ah-room --host "${HOST}" --port "${PORT}" --ready-timeout "${TIMEOUT}" \
  || { nemo services logs --instance ah-room -n 40 >&2; die "NeMo Helix did not start (nemo services logs --instance ah-room)"; }
# shellcheck source=/dev/null
. /usr/local/lib/ah-room/models.sh
ah_put_inference_secret "${HOME}/.agent-hardener/.env" || die "could not store the key as the NeMo Secret agent-hardener-inference-key"
printf 'ah-room: the key is the NeMo Secret agent-hardener-inference-key\n'
if [[ "${HOST}" == 127.0.0.1 && "${AH_ROOM_DOCKER:-private}" != host ]]; then
  printf 'ah-room: Studio is closed; to open it, start the room with -p 127.0.0.1:8080:8080 -e NHX_ROOM_HOST=0.0.0.0\n'
else
  printf 'ah-room: Studio: %s\n' "${AH_ROOM_STUDIO_URL:-http://localhost:${PORT}/studio}"
fi
