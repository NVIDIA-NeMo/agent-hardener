# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Register/connect the local OpenShell ``auto-defender`` gateway with the Docker driver.

The Python home for what ``scripts/setup.sh`` did inline. Best-effort: steps tolerate failure and
:func:`register_local_gateway` returns a :class:`GatewayResult` rather than raising. Installing the
OpenShell CLI/service itself stays out of scope (needs brew/sudo).
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import tomli_w

from agent_hardener.openshell.support import _poll_until

# GATEWAY_NAME matches OpenShellConfig.gateway's default so `run` finds the same gateway.
GATEWAY_ENDPOINT = "https://127.0.0.1:17670"
GATEWAY_NAME = "auto-defender"
DRIVER = "docker"
DOCKER_NETWORK = "openshell-docker"
#: The OpenShell release this code targets: ``setup`` installs it and older releases are refused.
OPENSHELL_VERSION = "v0.1.6"
INSTALL_HINT = (
    "curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh "
    f"| OPENSHELL_VERSION={OPENSHELL_VERSION} sh"
)
#: First file the gateway service reads (before the Homebrew/package default). Schema v2.
GATEWAY_CONFIG = Path.home() / ".config" / "openshell" / "gateway.toml"
BREW_SERVICE = "nvidia/openshell/openshell"
GATEWAY_PORT = 17670
#: Names a sandbox's control supervisor can reach the macOS host by, in preference order. Docker Desktop
#: and OrbStack serve the first; Lima-based runtimes (Colima, Rancher Desktop) always serve the second
#: and only some releases the first.
HOST_ALIASES = ("host.docker.internal", "host.lima.internal")
_COLIMA_SOCKET = re.compile(r"/\.colima/(?P<profile>[^/]+)/docker\.sock$")
#: Bound on each setup/probe command, so a wedged Docker, brew or systemd never hangs `setup`.
SETUP_COMMAND_TIMEOUT_S = 30.0
SYSTEMD_USER_SERVICE = "openshell-gateway"
READY_TIMEOUT_S = 30


@dataclass
class GatewayResult:
    ok: bool
    message: str


def _run(
    cmd: list[str], *, stdin: str | None = None, timeout_s: float = SETUP_COMMAND_TIMEOUT_S
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False, timeout=timeout_s)


def docker_reachable(*, timeout_s: float = SETUP_COMMAND_TIMEOUT_S) -> bool:
    """True when the docker CLI is installed and its daemon answers ``docker info``."""
    if shutil.which("docker") is None:
        return False
    try:
        return _run(["docker", "info"], timeout_s=timeout_s).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _release(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def outdated_openshell(openshell_bin: str = "openshell", *, timeout_s: float = SETUP_COMMAND_TIMEOUT_S) -> str | None:
    """Why the installed OpenShell CLI is older than :data:`OPENSHELL_VERSION`, or None when it is recent enough."""
    try:
        output = _run([openshell_bin, "--version"], timeout_s=timeout_s).stdout
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"`{openshell_bin} --version` failed ({exc})"
    installed, required = _release(output), _release(OPENSHELL_VERSION)
    if installed is None:
        return f"could not read the OpenShell version from `{openshell_bin} --version` ({output.strip()!r})"
    if required is not None and installed < required:
        return f"OpenShell {'.'.join(map(str, installed))} is older than the required {OPENSHELL_VERSION}"
    return None


def gateway_connected(
    gateway: str = GATEWAY_NAME, *, openshell_bin: str = "openshell", timeout_s: float = SETUP_COMMAND_TIMEOUT_S
) -> bool:
    """True when ``openshell status`` reports ``gateway`` as connected."""
    try:
        proc = _run([openshell_bin, "status", "--gateway", gateway], timeout_s=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0 and "Connected" in proc.stdout


def local_compute_driver(path: Path = GATEWAY_CONFIG) -> str | None:
    """The compute driver the local gateway service runs sandboxes with, as ``setup`` configured it."""
    if not path.is_file():
        return None
    driver = (
        tomllib.loads(path.read_text(encoding="utf-8")).get("openshell", {}).get("gateway", {}).get("compute_driver")
    )
    return driver if isinstance(driver, str) else None


def docker_endpoint(*, timeout_s: float = SETUP_COMMAND_TIMEOUT_S) -> str | None:
    """The daemon the docker CLI talks to now: ``DOCKER_HOST``, else the active ``docker context``.

    Runtime-agnostic: Docker Desktop, Colima (any profile), OrbStack, Rancher Desktop or a Linux dockerd
    each register their socket as a context.
    """
    if shutil.which("docker") is None:
        return None
    try:
        proc = _run(["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], timeout_s=timeout_s)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def host_alias(socket: str) -> str:
    """The host alias the runtime behind ``socket`` resolves, for sandboxes to reach the gateway by.

    Colima is asked from inside its VM (the first alias that resolves wins); other runtimes serve
    ``host.docker.internal``.
    """
    match = _COLIMA_SOCKET.search(socket)
    if match is None:
        return HOST_ALIASES[0]
    command = ["colima", "ssh", "--profile", match["profile"], "--", "getent", "hosts", *HOST_ALIASES]
    try:
        proc = _run(command)
    except (OSError, subprocess.TimeoutExpired):
        return HOST_ALIASES[0]
    resolved = {name for line in proc.stdout.splitlines() for name in line.split()[1:]}
    return next((alias for alias in HOST_ALIASES if alias in resolved), HOST_ALIASES[0])


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
        docker["grpc_endpoint"] = f"https://{host_alias(socket)}:{GATEWAY_PORT}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(config), encoding="utf-8")


def _step_error(cmd: list[str], what: str) -> str | None:
    """Run one setup step; ``None`` on success, else why it failed (with the command's stderr)."""
    try:
        proc = _run(cmd)
    except OSError as exc:
        return f"{what} failed: {exc}"
    except subprocess.TimeoutExpired:
        return f"{what} timed out after {SETUP_COMMAND_TIMEOUT_S:g}s"
    if proc.returncode == 0:
        return None
    detail = (proc.stderr or proc.stdout).strip()
    return f"{what} failed (exit {proc.returncode}){': ' + detail if detail else ''}"


def _restart_gateway_service() -> str | None:
    """Restart the gateway service so it reads the new config; a warning when that could not be done."""
    if sys.platform == "darwin":
        error = _step_error(["brew", "services", "restart", BREW_SERVICE], f"restarting {BREW_SERVICE}")
    elif shutil.which("systemctl") is not None:
        error = _step_error(
            ["systemctl", "--user", "restart", SYSTEMD_USER_SERVICE],
            f"restarting the {SYSTEMD_USER_SERVICE} user service",
        )
    else:
        return f"systemd not found — restart the OpenShell gateway so it picks up {GATEWAY_CONFIG}, then re-run."
    return f"{error} — restart it, then re-run." if error else None


def _ensure_docker_network() -> str | None:
    """Create the sandbox network the Linux Docker driver attaches to; why not, when that fails."""
    if _step_error(["docker", "network", "inspect", DOCKER_NETWORK], "inspecting the Docker network") is None:
        return None
    return _step_error(["docker", "network", "create", DOCKER_NETWORK], f"creating Docker network {DOCKER_NETWORK}")


def _wait_connected(timeout_s: int = READY_TIMEOUT_S) -> bool:
    return _poll_until(gateway_connected, timeout=timeout_s, interval=1.0)


def _missing_prerequisite() -> str | None:
    if shutil.which("openshell") is None:
        return f"OpenShell CLI not found — install it: {INSTALL_HINT}"
    if outdated := outdated_openshell():
        return f"{outdated} — upgrade it: {INSTALL_HINT}"
    if not docker_reachable():
        return "Docker daemon not reachable — start Docker, then re-run."
    return None


def register_local_gateway(*, wait: bool = True) -> GatewayResult:
    """Configure the Docker driver and register the local ``auto-defender`` gateway (idempotent)."""
    if missing := _missing_prerequisite():
        return GatewayResult(ok=False, message=missing)
    socket = docker_endpoint()
    if socket is None:
        return GatewayResult(ok=False, message="Could not resolve the Docker socket via `docker context inspect`.")

    write_gateway_config(socket)
    restart_warning = _restart_gateway_service()
    if sys.platform != "darwin" and (network_error := _ensure_docker_network()):
        return GatewayResult(ok=False, message=network_error)
    # Non-fatal: re-adding an already registered gateway fails, and the connect check below decides.
    add_warning = _step_error(
        ["openshell", "gateway", "add", GATEWAY_ENDPOINT, "--local", "--name", GATEWAY_NAME],
        f"registering gateway '{GATEWAY_NAME}'",
    )
    if wait and not _wait_connected():
        hint = f"Gateway not connected yet — check: openshell status --gateway {GATEWAY_NAME}"
        return GatewayResult(ok=False, message=" ".join(w for w in (restart_warning, add_warning, hint) if w))
    # A connected gateway proves the registration; a failed restart may still leave it on the old config.
    ready = f"OpenShell gateway '{GATEWAY_NAME}' ready."
    return GatewayResult(ok=True, message=f"{ready} {restart_warning}" if restart_warning else ready)
