# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Before the sandbox is built, every host tool the run will shell out to must be there and answer.

A missing OpenShell CLI, a stopped Docker daemon, a disconnected gateway, an absent compose file or an
unprovisioned garak venv otherwise surfaces minutes in, as a failed build or attacker. Each check reuses
the code the run itself resolves that tool with, so the preflight cannot disagree with the run.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from agent_hardener.agents.attackers.agent_breaker.run_agent_breaker import garak_base_command
from agent_hardener.env import GARAK_COMMAND
from agent_hardener.errors import ConfigError, SandboxProvisioningError
from agent_hardener.garak_venv import GARAK_PYTHON_ENVVAR
from agent_hardener.manifest import AGENT_BREAKER_IMPLEMENTATION
from agent_hardener.openshell.gateway import DRIVER as DOCKER_DRIVER
from agent_hardener.openshell.gateway import GATEWAY_NAME, docker_reachable, gateway_connected, local_compute_driver
from agent_hardener.providers.backends import compose_file_path, resolve_compose_command
from agent_hardener.runtime.runner import recorded_victim_image
from agent_hardener.tools.openshell import openshell_config, sandbox_is_ready

if TYPE_CHECKING:
    from agent_hardener.models import BackendServiceSpec, SessionConfig
    from agent_hardener.openshell.config import OpenShellConfig

#: Bound on each host probe (``docker info``, ``openshell status``): a wedged daemon must fail the
#: preflight, not hang it.
PROBE_TIMEOUT_S = 15.0
SETUP_HINT = "Run `agent-hardener setup`, then retry."


def preflight_host(session: SessionConfig, *, reuse: bool = False) -> None:
    """Raise a classified :class:`~agent_hardener.errors.AgentHardenerError` for the first missing host prerequisite.

    ``reuse`` is the run's ``--reuse``: a Ready sandbox on a recorded image is reused without a build.
    """
    if session.victim_control.type == "openshell":
        _require_openshell(session, openshell_config(session.victim_control), reuse=reuse)
    if any(attacker.implementation == AGENT_BREAKER_IMPLEMENTATION for attacker in session.attackers):
        _require_garak()


def _require_openshell(session: SessionConfig, config: OpenShellConfig, *, reuse: bool) -> None:
    if shutil.which(config.openshell_bin) is None:
        raise SandboxProvisioningError(
            f"OpenShell CLI not found: {config.openshell_bin!r} is not on PATH.",
            remediation=f"Install the OpenShell CLI, or set openshell_bin to its path. {SETUP_HINT}",
        )
    if not gateway_connected(config.gateway, openshell_bin=config.openshell_bin, timeout_s=PROBE_TIMEOUT_S):
        raise SandboxProvisioningError(
            f"OpenShell gateway '{config.gateway}' is not connected.",
            remediation=f"Check `openshell status --gateway {config.gateway}`. {SETUP_HINT}",
        )
    composed = [spec for spec in (config.relay_victim.backends if config.relay_victim else []) if spec.compose_file]
    needs_docker = bool(composed) or _runs_on_local_docker(config) or _builds_locally(session, config, reuse=reuse)
    if needs_docker and not docker_reachable(timeout_s=PROBE_TIMEOUT_S):
        raise SandboxProvisioningError(
            "Docker daemon not reachable: `docker info` failed.",
            remediation="Start Docker (or point DOCKER_HOST at a running daemon), then retry.",
        )
    if composed:
        _require_compose(composed, config.cwd or Path.cwd())


def _runs_on_local_docker(config: OpenShellConfig) -> bool:
    """The local gateway ``setup`` registers runs every sandbox as a container on this host's Docker."""
    return config.gateway == GATEWAY_NAME and local_compute_driver() == DOCKER_DRIVER


def _builds_locally(session: SessionConfig, config: OpenShellConfig, *, reuse: bool) -> bool:
    """Whether the run will ``docker build`` the victim here: no pre-built image, and no Ready sandbox to reuse."""
    if config.image is not None:
        return False
    reusable = reuse and recorded_victim_image(session, config) is not None and sandbox_is_ready(config)
    return not reusable


def _require_compose(specs: list[BackendServiceSpec], cwd: Path) -> None:
    resolve_compose_command()
    for spec in specs:
        path = compose_file_path(spec, cwd)
        if not path.is_file():
            raise ConfigError(
                f"Compose file for backend '{spec.name}' not found: {path}",
                remediation="Fix the backend's compose_file in the manifest (a relative path is taken from cwd).",
            )


def _require_garak() -> None:
    command = garak_base_command()
    if shutil.which(command[0]) is None:
        raise ConfigError(
            f"garak command not found: {command[0]!r} is not an executable.",
            remediation=f"Fix ${GARAK_COMMAND} or ${GARAK_PYTHON_ENVVAR}. {SETUP_HINT}",
        )
