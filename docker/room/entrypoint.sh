#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Start the room: private dockerd, OpenShell gateway `auto-defender`, the inference key, then
# every executable in /etc/ah-room/start.d (an image built on this one adds its services there).
# Any failure stops the container with one "ah-room:" line saying what to change.
#
# AH_ROOM_DOCKER=host uses the host's Docker Engine instead of a private one: no --privileged, but
# the host's socket mounted at /var/run/docker.sock and --network host, because OpenShell's
# supervisor containers run on the host network and call the gateway on 127.0.0.1.
#
# AH_ROOM_OPENSHELL=host (with AH_ROOM_DOCKER=host) starts no gateway either: the room uses the one
# already registered on the host as `auto-defender`, whose ~/.config/openshell is mounted read-only
# at /run/host-openshell and copied, so nothing the room does writes to the host's registration.
#
# The keys are never exported: they are written to agent-hardener's operator dotenv (mode 600), which
# agent-hardener and its Helix plugin read themselves, so the room's services do not inherit them.
set -euo pipefail

TLS_DIR="${HOME}/.local/state/openshell/tls"
KEY_SRC="${AH_KEYS_FILE:-/run/secrets/keys.env}"
OPERATOR_ENV="${HOME}/.agent-hardener/.env"
DOCKER_MODE="${AH_ROOM_DOCKER:-private}"
OPENSHELL_MODE="${AH_ROOM_OPENSHELL:-room}"
HOST_OPENSHELL=/run/host-openshell
# agent-hardener looks for the gateway under this name at this endpoint.
GATEWAY_NAME=auto-defender
GATEWAY_PORT=17670
GATEWAY_CONFIG="${HOME}/.config/openshell/gateway.toml"
# The network agent-hardener's setup creates for sandboxes on the Docker driver.
SANDBOX_NETWORK=openshell-docker
OPENSHELL_TAG="${OPENSHELL_VERSION#v}"
SUPERVISOR_IMAGE="${OPENSHELL_SUPERVISOR_IMAGE:-ghcr.io/nvidia/openshell/supervisor:${OPENSHELL_TAG}}"
RUNTIME_IMAGE="${OPENSHELL_SANDBOX_RUNTIME_IMAGE:-ghcr.io/nvidia/openshell/sandbox:${OPENSHELL_TAG}}"
DOCKER_TMPFS="${AH_ROOM_DOCKER_TMPFS:-12g}"
READY_FILE=/run/ah-room/ready
# NeMo Helix keeps its database, ClickHouse, encryption key and files here.
HELIX_DATA="${HOME}/.local/share/nemo"

die() {
  printf 'ah-room: %s\n' "$*" >&2
  exit 1
}

check_runtime() {
  [[ "${DOCKER_MODE}" == private || "${DOCKER_MODE}" == host ]] \
    || die "AH_ROOM_DOCKER=${DOCKER_MODE}: use private (the default) or host"
  [[ "${OPENSHELL_MODE}" == room || "${OPENSHELL_MODE}" == host ]] \
    || die "AH_ROOM_OPENSHELL=${OPENSHELL_MODE}: use room (the default) or host"
  if [[ "${OPENSHELL_MODE}" == host ]]; then
    [[ "${DOCKER_MODE}" == host ]] \
      || die "AH_ROOM_OPENSHELL=host needs AH_ROOM_DOCKER=host: the host's gateway runs sandboxes on the host's Docker"
    [[ -d "${HOST_OPENSHELL}/gateways/${GATEWAY_NAME}" ]] \
      || die "no gateway ${GATEWAY_NAME} in ${HOST_OPENSHELL}; mount the host's ~/.config/openshell there (agent-hardener setup registers it)"
  fi
  [[ "$(id -u)" == "0" ]] || die "the room must run as root (drop --user); it starts its own dockerd"
  mkdir -p "${HOME}/.agent-hardener" /var/lib/docker 2>/dev/null \
    || die "the root filesystem is read-only; drop --read-only"
  mkdir -p /run/ah-room
  rm -f "${READY_FILE}"
  if [[ "${DOCKER_MODE}" == host ]]; then
    [[ -S /var/run/docker.sock ]] \
      || die "AH_ROOM_DOCKER=host needs the host's Docker socket: -v /var/run/docker.sock:/var/run/docker.sock"
    return 0
  fi
  local probe
  probe="$(mktemp -d)"
  if ! mount -t tmpfs -o size=1m tmpfs "${probe}" 2>/dev/null; then
    rmdir "${probe}"
    die "cannot mount inside the container; start it with --privileged"
  fi
  umount "${probe}"
  rmdir "${probe}"
}

ensure_sandbox_user() {
  # The sandbox archive is copied as uid:gid 999 and dockerd resolves it on this container.
  if ! getent group 999 >/dev/null; then
    groupadd --gid 999 sandbox
  fi
  if ! getent passwd 999 >/dev/null; then
    useradd --uid 999 --gid 999 --system --create-home --home-dir /home/sandbox --shell /usr/sbin/nologin sandbox
  fi
}

prepare_docker_data() {
  # The container's own filesystem is overlay, and the kernel rejects overlay on overlay, so the
  # inner engine needs a real filesystem: a mounted volume if one is given, else a tmpfs.
  mkdir -p /var/lib/docker
  if mountpoint -q /var/lib/docker; then
    return 0
  fi
  [[ "${DOCKER_TMPFS}" != "off" ]] \
    || die "AH_ROOM_DOCKER_TMPFS=off needs a volume mounted at /var/lib/docker"
  mount -t tmpfs -o "size=${DOCKER_TMPFS}" tmpfs /var/lib/docker
}

configure_docker_proxy() {
  # dockerd reads the proxy from its environment; image builds read it from the client config.
  local http="${HTTP_PROXY:-${http_proxy:-}}"
  local https="${HTTPS_PROXY:-${https_proxy:-}}"
  local no="${NO_PROXY:-${no_proxy:-}}"
  [[ -n "${http}${https}" ]] || return 0
  mkdir -p "${HOME}/.docker"
  python3 - "${HOME}/.docker/config.json" "${http}" "${https}" "${no}" <<'PY'
import json
import sys
from pathlib import Path

path, http, https, no = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
config = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
proxy = {key: value for key, value in (("httpProxy", http), ("httpsProxy", https), ("noProxy", no)) if value}
config["proxies"] = {"default": proxy}
path.write_text(json.dumps(config, indent=2), encoding="utf-8")
PY
}

use_host_docker() {
  docker info >/dev/null 2>&1 \
    || die "cannot reach the host's Docker Engine through /var/run/docker.sock: $(docker info 2>&1 | tail -n 1)"
  if [[ "${OPENSHELL_MODE}" == room ]]; then
    ! gateway_listening \
      || die "port ${GATEWAY_PORT} is taken on this host (another OpenShell gateway?); stop it, use it with --host-openshell, or run the room without --host-docker"
  fi
  configure_docker_proxy
}

wait_connected() {
  local seconds="$1" _i
  for _i in $(seq 1 "${seconds}"); do
    if openshell status --gateway "${GATEWAY_NAME}" 2>/dev/null | grep -q Connected; then
      return 0
    fi
    sleep 1
  done
  return 1
}

use_host_gateway() {
  mkdir -p "${HOME}/.config"
  cp -r "${HOST_OPENSHELL}" "${HOME}/.config/openshell"
  gateway_listening \
    || die "this host's OpenShell gateway does not answer on 127.0.0.1:${GATEWAY_PORT}; start it (openshell status), or run without --host-openshell"
  wait_connected 15 \
    || die "this host's gateway ${GATEWAY_NAME} is not Connected (openshell status --gateway ${GATEWAY_NAME})"
}

start_dockerd() {
  ensure_sandbox_user
  if [[ "${DOCKER_MODE}" == host ]]; then
    use_host_docker
    return 0
  fi
  if docker info >/dev/null 2>&1; then
    return 0
  fi
  prepare_docker_data
  configure_docker_proxy
  mkdir -p /var/log /var/run
  # shellcheck disable=SC2086
  dockerd --host=unix:///var/run/docker.sock --iptables=true ${AH_ROOM_DOCKERD_ARGS:-} >/var/log/dockerd.log 2>&1 &
  local _i
  for _i in $(seq 1 90); do
    if docker info >/dev/null 2>&1; then
      remove_stale_containers
      return 0
    fi
    sleep 1
  done
  tail -n 40 /var/log/dockerd.log >&2 || true
  die "dockerd did not become ready (log: /var/log/dockerd.log)"
}

remove_stale_containers() {
  # A /var/lib/docker volume from an earlier room keeps its images as a cache, and also the containers
  # that room's gateway and Helix left; this engine is the room's own, so none of them is anyone else's.
  local stale
  stale="$(docker ps -aq)"
  [[ -n "${stale}" ]] || return 0
  printf 'ah-room: removing %s containers left by an earlier room\n' "$(wc -w <<<"${stale}")"
  # shellcheck disable=SC2086
  docker rm -f ${stale} >/dev/null
}

write_gateway_config() {
  # OpenShell's schema-v2 config with the keys `agent-hardener setup` sets on a host (the Docker
  # driver on this socket), plus the sandbox JWT keys, which a standalone gateway needs.
  mkdir -p "$(dirname "${GATEWAY_CONFIG}")"
  cat >"${GATEWAY_CONFIG}" <<EOF
[openshell]
version = 2

[openshell.gateway]
compute_driver = "docker"

[openshell.gateway.gateway_jwt]
signing_key_path = "${TLS_DIR}/jwt/signing.pem"
public_key_path = "${TLS_DIR}/jwt/public.pem"
kid_path = "${TLS_DIR}/jwt/kid"
gateway_id = "ah-room"
ttl_secs = 3600

[openshell.drivers.docker]
socket_path = "/var/run/docker.sock"
EOF
}

register_gateway() {
  docker network inspect "${SANDBOX_NETWORK}" >/dev/null 2>&1 || docker network create "${SANDBOX_NETWORK}" >/dev/null
  openshell gateway add "https://127.0.0.1:${GATEWAY_PORT}" --local --name "${GATEWAY_NAME}" >/dev/null
  wait_connected 30
}

install_client_cert() {
  local name="$1"
  local dest="${HOME}/.config/openshell/gateways/${name}/mtls"
  mkdir -p "${dest}"
  cp "${TLS_DIR}/ca.crt" "${dest}/ca.crt"
  cp "${TLS_DIR}/client/tls.crt" "${dest}/tls.crt"
  cp "${TLS_DIR}/client/tls.key" "${dest}/tls.key"
  chmod 600 "${dest}/tls.key"
}

gateway_listening() {
  python3 -c 'import socket, sys; socket.create_connection(("127.0.0.1", int(sys.argv[1])), 1).close()' \
    "${GATEWAY_PORT}" 2>/dev/null
}

start_gateway() {
  mkdir -p "${TLS_DIR}"
  if [[ ! -f "${TLS_DIR}/server/tls.crt" ]]; then
    openshell-gateway generate-certs --output-dir "${TLS_DIR}" --server-san 127.0.0.1 --server-san localhost
  fi
  [[ -f "${TLS_DIR}/server/tls.crt" && -f "${TLS_DIR}/server/tls.key" && -f "${TLS_DIR}/ca.crt" ]] \
    || die "gateway certificate generation did not write the server bundle"
  [[ -f "${TLS_DIR}/jwt/signing.pem" ]] || die "gateway certificate generation did not write the sandbox JWT key"
  write_gateway_config
  install_client_cert "${GATEWAY_NAME}"
  install_client_cert openshell
  # The gateway resolves the supervisor image to an id once, so the images must be present
  # before it starts; a missing one fails every sandbox with ControlSupervisorStartFailed.
  export OPENSHELL_SUPERVISOR_IMAGE="${SUPERVISOR_IMAGE}"
  export OPENSHELL_SANDBOX_RUNTIME_IMAGE="${RUNTIME_IMAGE}"
  local image
  for image in "${OPENSHELL_SUPERVISOR_IMAGE}" "${OPENSHELL_SANDBOX_RUNTIME_IMAGE}"; do
    docker pull --quiet "${image}" >/dev/null \
      || die "cannot pull ${image}: the room needs registry access (network, DNS or HTTPS_PROXY)"
  done
  openshell-gateway \
    --bind-address 127.0.0.1 \
    --port "${GATEWAY_PORT}" \
    --tls-cert "${TLS_DIR}/server/tls.crt" \
    --tls-key "${TLS_DIR}/server/tls.key" \
    --tls-client-ca "${TLS_DIR}/ca.crt" \
    --config "${GATEWAY_CONFIG}" \
    >/var/log/openshell-gateway.log 2>&1 &
  register_gateway || {
    tail -n 40 /var/log/openshell-gateway.log >&2 || true
    die "gateway ${GATEWAY_NAME} did not become Connected (log: /var/log/openshell-gateway.log)"
  }
}

load_keys() {
  [[ -f "${KEY_SRC}" ]] || die "no key file at ${KEY_SRC}; mount one with -v <keys.env>:${KEY_SRC}:ro"
  python3 /usr/local/lib/ah-room/keys.py "${KEY_SRC}" "${OPERATOR_ENV}"
  grep -q '^INFERENCE_API_KEY=.' "${OPERATOR_ENV}" || die "INFERENCE_API_KEY was not written to ${OPERATOR_ENV}"
}

run_start_hooks() {
  local hook
  for hook in /etc/ah-room/start.d/*; do
    [[ -e "${hook}" ]] || continue
    [[ -x "${hook}" ]] || die "start hook ${hook} is not executable"
    "${hook}" || die "start hook ${hook} failed"
  done
}

use_data_volume() {
  if [[ -z "${AH_ROOM_DATA:-}" ]]; then
    printf 'ah-room: no data volume (AH_ROOM_DATA); NeMo Helix data is lost when this container is removed\n'
    return 0
  fi
  [[ "${AH_ROOM_DATA}" == /* && -d "${AH_ROOM_DATA}" ]] \
    || die "AH_ROOM_DATA=${AH_ROOM_DATA} is not a mounted directory; mount a volume there"
  [[ ! -e "${HELIX_DATA}" || -L "${HELIX_DATA}" ]] || die "${HELIX_DATA} already exists in the image; it cannot move to the data volume"
  # Helix resolves this link, so with AH_ROOM_DOCKER=host the ClickHouse bind mount names the path
  # the volume has on the host: AH_ROOM_DATA must be that path there.
  mkdir -p "${AH_ROOM_DATA}/nemo" "$(dirname "${HELIX_DATA}")"
  ln -sfn "${AH_ROOM_DATA}/nemo" "${HELIX_DATA}"
}

check_runtime
use_data_volume
load_keys
start_dockerd
if [[ "${OPENSHELL_MODE}" == host ]]; then
  use_host_gateway
else
  start_gateway
fi
run_start_hooks
touch "${READY_FILE}"
printf 'ah-room: ready. Run the cycle with: ah-cycle --help\n'
exec "$@"
