# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register/connect the local OpenShell ``auto-defender`` gateway with the Docker driver.

The Python home for what ``scripts/setup.sh`` did inline. Best-effort: steps tolerate failure and
:func:`register_local_gateway` returns a :class:`GatewayResult` rather than raising. Installing the
OpenShell CLI/service itself stays out of scope (needs brew/sudo).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

# GATEWAY_NAME matches OpenShellConfig.gateway's default so `run` finds the same gateway.
GATEWAY_ENDPOINT = "https://127.0.0.1:17670"
GATEWAY_NAME = "auto-defender"
DRIVER = "docker"
DOCKER_NETWORK = "openshell-docker"
#: The OpenShell release this code targets.
OPENSHELL_VERSION = "v0.1.4"
INSTALL_HINT = (
    "curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh "
    f"| OPENSHELL_VERSION={OPENSHELL_VERSION} sh"
)
#: First file the gateway service reads (before the Homebrew/package default). Schema v2.
GATEWAY_CONFIG = Path.home() / ".config" / "openshell" / "gateway.toml"
BREW_SERVICE = "nvidia/openshell/openshell"
DOCKER_DESKTOP_GRPC_ENDPOINT = "https://host.docker.internal:17670"
SYSTEMD_USER_SERVICE = "openshell-gateway"
READY_TIMEOUT_S = 30


@dataclass
class GatewayResult:
    ok: bool
    message: str


def _run(cmd: list[str], *, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False)


def _docker_reachable() -> bool:
    return shutil.which("docker") is not None and _run(["docker", "info"]).returncode == 0


def _docker_socket() -> str | None:
    sock = _run(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"]).stdout.strip()
    return sock or None


def write_gateway_config(socket: str, path: Path = GATEWAY_CONFIG) -> None:
    """Point the gateway at the Docker driver and this daemon's socket in its schema-v2 ``gateway.toml``.

    Existing keys in the file are preserved; only the driver keys below are set.
    """
    config = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    openshell = config.setdefault("openshell", {})
    openshell["version"] = 2
    openshell.setdefault("gateway", {})["compute_driver"] = DRIVER
    docker = openshell.setdefault("drivers", {}).setdefault(DRIVER, {})
    docker["socket_path"] = socket.removeprefix("unix://")
    if sys.platform == "darwin":
        # The driver runs each sandbox's control supervisor with host networking. Under Docker Desktop
        # (without its opt-in host networking) "host" is the Linux VM, so the gateway's 127.0.0.1
        # listener is unreachable and every sandbox fails with ControlSupervisorStartFailed; the
        # Desktop-provided host alias does reach it.
        docker["grpc_endpoint"] = DOCKER_DESKTOP_GRPC_ENDPOINT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config), encoding="utf-8")


def _configure_driver_macos(socket: str) -> str | None:
    write_gateway_config(socket)
    _run(["brew", "services", "restart", BREW_SERVICE])
    return None


def _configure_driver_linux(socket: str) -> str | None:
    write_gateway_config(socket)
    warning: str | None = None
    if shutil.which("systemctl") is not None:
        restart = _run(["systemctl", "--user", "restart", SYSTEMD_USER_SERVICE])
        if restart.returncode != 0:
            warning = f"could not restart the {SYSTEMD_USER_SERVICE} user service — restart it, then re-run."
    else:
        warning = f"systemd not found — restart the OpenShell gateway so it picks up {GATEWAY_CONFIG}, then re-run."
    if _run(["docker", "network", "inspect", DOCKER_NETWORK]).returncode != 0:
        _run(["docker", "network", "create", DOCKER_NETWORK])
    return warning


def _wait_connected(timeout_s: int = READY_TIMEOUT_S) -> bool:
    for _ in range(timeout_s):
        proc = _run(["openshell", "status", "--gateway", GATEWAY_NAME])
        if proc.returncode == 0 and "Connected" in proc.stdout:
            return True
        time.sleep(1)
    return False


def register_local_gateway(*, wait: bool = True) -> GatewayResult:
    """Configure the Docker driver and register the local ``auto-defender`` gateway (idempotent)."""
    if shutil.which("openshell") is None:
        return GatewayResult(ok=False, message=f"OpenShell CLI not found — install it: {INSTALL_HINT}")
    if not _docker_reachable():
        return GatewayResult(ok=False, message="Docker daemon not reachable — start Docker, then re-run.")
    socket = _docker_socket()
    if socket is None:
        return GatewayResult(ok=False, message="Could not resolve the Docker socket via `docker context inspect`.")

    configure = _configure_driver_macos if sys.platform == "darwin" else _configure_driver_linux
    warning = configure(socket)

    _run(["openshell", "gateway", "add", GATEWAY_ENDPOINT, "--local", "--name", GATEWAY_NAME])

    if wait and not _wait_connected():
        hint = f"Gateway not connected yet — check: openshell status --gateway {GATEWAY_NAME}"
        return GatewayResult(ok=False, message=f"{warning} {hint}" if warning else hint)
    return GatewayResult(ok=True, message=f"OpenShell gateway '{GATEWAY_NAME}' ready.")
