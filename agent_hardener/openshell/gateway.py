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
from dataclasses import dataclass

# GATEWAY_NAME matches OpenShellConfig.gateway's default so `run` finds the same gateway.
GATEWAY_ENDPOINT = "https://127.0.0.1:17670"
GATEWAY_NAME = "auto-defender"
DRIVER = "docker"
DOCKER_NETWORK = "openshell-docker"
SYSTEMD_DROPIN_DIR = "/etc/systemd/system/openshell.service.d"
SYSTEMD_DROPIN = f"{SYSTEMD_DROPIN_DIR}/docker.conf"
INSTALL_HINT = "curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh"
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


def _configure_driver_macos(socket: str) -> str | None:
    _run(["brew", "services", "stop", "openshell"])
    _run(["launchctl", "setenv", "OPENSHELL_DRIVERS", DRIVER])
    _run(["launchctl", "setenv", "DOCKER_HOST", socket])
    _run(["brew", "services", "start", "openshell"])
    return None


def _configure_driver_linux(socket: str) -> str | None:
    have_service = (
        shutil.which("systemctl") is not None
        and "openshell" in _run(["systemctl", "list-units", "--all", "--type=service"]).stdout
    )
    warning: str | None = None
    if have_service:
        _run(["sudo", "systemctl", "stop", "openshell"])
        _run(["sudo", "mkdir", "-p", SYSTEMD_DROPIN_DIR])
        dropin = f"[Service]\nEnvironment=OPENSHELL_DRIVERS={DRIVER}\nEnvironment=DOCKER_HOST={socket}\n"
        _run(["sudo", "tee", SYSTEMD_DROPIN], stdin=dropin)
        _run(["sudo", "systemctl", "daemon-reload"])
        _run(["sudo", "systemctl", "start", "openshell"])
    else:
        warning = (
            f"systemd openshell service not found — set OPENSHELL_DRIVERS={DRIVER} and "
            f"DOCKER_HOST={socket} for the OpenShell service, then re-run."
        )
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
