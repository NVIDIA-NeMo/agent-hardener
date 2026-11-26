#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the Agent Hardener room from agent-hardener and NeMo Helix.
#
#   docker/build.sh [--ah-ref REF | --agent-hardener DIR] [--helix-ref REF | --helix DIR]
#                   [--ah-repo URL] [--helix-repo URL] [--tag IMG] [--platform linux/amd64|linux/arm64]
#
# With no source flags, both are fetched at main: NVIDIA-NeMo/agent-hardener and NVIDIA-NeMo/nemo-helix.
# --ah-ref / --helix-ref fetch a branch, tag or commit instead (--*-repo for a fork), into
# $AH_ROOM_SRC (~/.cache/ah-room/src). --agent-hardener / --helix build a local checkout as it is
# (--agent-hardener . for this one). The room's own files always come from this checkout's docker/.
#
# Three images, in order:
#   agent-hardener-room:<version>   agent-hardener, garak, OpenShell and Docker (room/Dockerfile)
#   nhx-studio-ui:dev               the Studio UI assets (from the Helix checkout)
#   IMG                             NeMo Helix on top: the image users run (nemo-agent-hardener-room:dev)
# <version> is the agent-hardener version in its pyproject.toml. Needs docker buildx, git and python3.
# IMG records what it was built from, when and by whom (AH_ROOM_BUILT_BY, else git's user.email):
# labels (docker inspect) and /usr/local/share/ah-room/versions.json (ah-room version).
# --platform builds for another CPU (QEMU emulation, a few hours for arm64; see the README). The room
# and IMG then get an -<arch> suffix (agent-hardener-room:0.0.19-arm64), so both CPUs can sit side by side.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
AH=""
AH_REPO=https://github.com/NVIDIA-NeMo/agent-hardener.git
AH_REF=""
HELIX=""
HELIX_REPO=https://github.com/NVIDIA-NeMo/nemo-helix.git
HELIX_REF=""
TAG=nemo-agent-hardener-room:dev
PLATFORM=""
SRC="${AH_ROOM_SRC:-${HOME}/.cache/ah-room/src}"

die() {
  printf 'build: %s\n' "$*" >&2
  exit 1
}

usage() {
  awk 'NR >= 5 && !/^#/ { exit } NR >= 5 { sub(/^# ?/, ""); print }' "$0"
}

# fetch NAME URL REF: check REF of URL out into $SRC/NAME, replacing what was there.
fetch() {
  local dir="${SRC}/$1"
  [[ -d "${dir}/.git" ]] || git init -q "${dir}"
  printf 'build: fetching %s %s\n' "$2" "$3" >&2
  git -C "${dir}" fetch -q --depth 1 "$2" "$3" || die "cannot fetch ${3} from ${2}"
  git -C "${dir}" checkout -q --detach -f FETCH_HEAD
  git -C "${dir}" clean -q -ffdx
  printf '%s\n' "${dir}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-hardener) AH="$2"; shift 2 ;;
    --ah-ref) AH_REF="$2"; shift 2 ;;
    --ah-repo) AH_REPO="$2"; shift 2 ;;
    --helix) HELIX="$2"; shift 2 ;;
    --helix-ref) HELIX_REF="$2"; shift 2 ;;
    --helix-repo) HELIX_REPO="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --platform) PLATFORM="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1 (see docker/build.sh --help)" ;;
  esac
done
[[ -z "${AH}" || -z "${AH_REF}" ]] || die "pass --agent-hardener DIR or --ah-ref REF, not both"
[[ -z "${HELIX}" || -z "${HELIX_REF}" ]] || die "pass --helix DIR or --helix-ref REF, not both"
docker buildx version >/dev/null 2>&1 || die "docker buildx is required (Docker Engine 23.0+, or install the buildx plugin)"
command -v git >/dev/null || die "git is required"

# Fetched sources are recorded with the repository and ref asked for; local ones with their own.
if [[ -n "${AH}" ]]; then
  AH="$(cd "${AH}" && pwd)"
  AH_REPO=""
else
  AH_REF="${AH_REF:-main}"
  AH="$(fetch agent-hardener "${AH_REPO}" "${AH_REF}")"
fi
if [[ -n "${HELIX}" ]]; then
  HELIX="$(cd "${HELIX}" && pwd)"
  HELIX_REPO=""
else
  HELIX_REF="${HELIX_REF:-main}"
  HELIX="$(fetch nemo-helix "${HELIX_REPO}" "${HELIX_REF}")"
fi
[[ -f "${AH}/pyproject.toml" ]] || die "${AH} is not an agent-hardener checkout (no pyproject.toml)"
[[ -f "${HELIX}/plugins/nemo-agent-hardener/pyproject.toml" ]] \
  || die "${HELIX} is not a NeMo Helix checkout with the nemo-agent-hardener plugin"
[[ -f "${HELIX}/docker/base/Dockerfile.nhx-studio-ui" ]] || die "${HELIX} has no docker/base/Dockerfile.nhx-studio-ui"

VERSION="$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["project"]["version"])' \
  "${AH}/pyproject.toml")"
[[ -n "${VERSION}" ]] || die "no project.version in ${AH}/pyproject.toml"
BASE="agent-hardener-room:${VERSION}"

NATIVE="linux/$(docker version --format '{{.Server.Arch}}')"
[[ "${NATIVE}" != "linux/" ]] || die "cannot read the Docker Engine's CPU architecture (docker version)"
PLATFORM="${PLATFORM:-${NATIVE}}"
case "${PLATFORM}" in
  linux/amd64|linux/arm64) ;;
  *) die "--platform ${PLATFORM}: only linux/amd64 and linux/arm64 are supported" ;;
esac
if [[ "${PLATFORM}" != "${NATIVE}" ]]; then
  BUILDER="$(docker buildx inspect --bootstrap)" || die "docker buildx inspect failed"
  [[ "${BUILDER}" == *"${PLATFORM}"* ]] \
    || die "this builder cannot build ${PLATFORM}; install QEMU first: docker run --privileged --rm tonistiigi/binfmt --install ${PLATFORM#linux/}"
  BASE="${BASE}-${PLATFORM#linux/}"
  TAG="${TAG}-${PLATFORM#linux/}"
fi

RECORD="$(python3 "${HERE}/helix-room/versions.py" build --room "$(git -C "${HERE}" rev-parse --show-toplevel)" --agent-hardener "${AH}" \
  --ah-repo "${AH_REPO}" --ah-ref "${AH_REF}" --helix "${HELIX}" --helix-repo "${HELIX_REPO}" \
  --helix-ref "${HELIX_REF}" --platform "${PLATFORM}" --image "${TAG}")"
LABELS=()
while IFS= read -r label; do
  LABELS+=(--label "${label}")
done < <(python3 "${HERE}/helix-room/versions.py" labels "${RECORD}")
[[ ${#LABELS[@]} -gt 0 ]] || die "versions.py printed no labels"

printf 'build: %s (%s) from %s\n' "${BASE}" "${PLATFORM}" "${AH}"
docker buildx build --load --platform "${PLATFORM}" -f "${HERE}/room/Dockerfile" --build-context ahd="${HERE}" \
  -t "${BASE}" "${AH}"

printf 'build: nhx-studio-ui:dev from %s\n' "${HELIX}"
docker buildx build --load -f "${HELIX}/docker/base/Dockerfile.nhx-studio-ui" --target artifacts \
  -t nhx-studio-ui:dev "${HELIX}"

printf 'build: %s (%s) from %s\n' "${TAG}" "${PLATFORM}" "${HELIX}"
docker buildx build --load --platform "${PLATFORM}" -f "${HERE}/helix-room/Dockerfile" --build-context ahd="${HERE}" \
  --build-arg "AH_ROOM_IMAGE=${BASE}" --build-arg "AH_ROOM_BUILD=${RECORD}" "${LABELS[@]}" -t "${TAG}" "${HELIX}"

"${HERE}/helix-room/ah-room" version --image "${TAG}"
printf 'build: done: %s\n' "${TAG}"
