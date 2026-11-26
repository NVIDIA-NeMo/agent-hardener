#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke test of a built room image, with no model key and no model call:
#
#   docker/smoke.sh IMG
#
# Through the launcher the image ships (ah-room), in private mode:
#   1. up: the room starts, Helix is ready, and the key is the NeMo Secret agent-hardener-inference-key;
#   2. no process in the room has the key in its environment;
#   3. down then up again: a NeMo Secret made before down is still there (the data volumes);
#   4. down --purge leaves no volume; --bind is refused.
set -euo pipefail

IMAGE="${1:-}"
NAME="ah-smoke-$$"

die() {
  printf 'smoke: FAIL: %s\n' "$*" >&2
  exit 1
}
pass() { printf 'smoke: ok: %s\n' "$*"; }

[[ -n "${IMAGE}" ]] || die "usage: docker/smoke.sh IMG"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || die "no image ${IMAGE} here"

work="$(mktemp -d)"
launcher="${work}/ah-room"
cleanup() {
  "${launcher}" down --purge --name "${NAME}" >/dev/null 2>&1 || true
  rm -rf "${work}"
}
trap cleanup EXIT
docker run --rm --entrypoint cat "${IMAGE}" /usr/local/share/ah-room/ah-room >"${launcher}"
chmod +x "${launcher}"
# A placeholder: nothing here calls a model.
printf 'INFERENCE_API_KEY=smoke-placeholder-key\n' >"${work}/keys.env"  # pragma: allowlist secret
chmod 600 "${work}/keys.env"
export AH_ROOM_IMAGE="${IMAGE}" AH_ROOM_NAME="${NAME}"
in_room() { docker exec "${NAME}" bash -c "cd /opt/nemo-helix && $1"; }

"${launcher}" up --keys "${work}/keys.env" --no-open
pass "up"
in_room 'nemo secrets get agent-hardener-inference-key --workspace default >/dev/null' \
  || die "the NeMo Secret agent-hardener-inference-key is missing"
pass "the key is the NeMo Secret agent-hardener-inference-key"

exposed="$(docker exec "${NAME}" bash -c '
  n=0
  for e in /proc/[0-9]*/environ; do
    tr "\0" "\n" <"$e" 2>/dev/null | grep -q "^INFERENCE_API_KEY=" && n=$((n + 1))
  done
  echo "$n"')"
[[ "${exposed}" == 0 ]] || die "${exposed} processes in the room have INFERENCE_API_KEY in their environment"
pass "no process has the key in its environment"

in_room 'printf kept | nemo secrets create ah-smoke-marker --from-file - --workspace default >/dev/null'
"${launcher}" down --name "${NAME}"
"${launcher}" up --keys "${work}/keys.env" --no-open
in_room 'nemo secrets get ah-smoke-marker --workspace default >/dev/null' \
  || die "a NeMo Secret made before down was lost on up"
pass "Helix's data survived down and up"

"${launcher}" down --purge --name "${NAME}"
left="$(docker volume ls -q --filter "label=ah-room.room=${NAME}")"
[[ -z "${left}" ]] || die "down --purge left volumes: ${left}"
pass "down --purge removed the volumes"

if "${launcher}" up --keys "${work}/keys.env" --bind 0.0.0.0 >/dev/null 2>&1; then
  die "--bind 0.0.0.0 was accepted"
fi
pass "--bind is refused"
printf 'smoke: all checks passed for %s\n' "${IMAGE}"
