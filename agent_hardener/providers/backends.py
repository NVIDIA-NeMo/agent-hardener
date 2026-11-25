# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side backend orchestration for victim agents.

An agent's tools often call a backend service (FastAPI, Postgres, …). Per the topology
decision, those backends run on the host *outside* OpenShell (the OpenShell sandbox hardens only
the agent). This module brings the declared docker-compose stacks up/down on the host; it has
nothing to do with the OpenShell gateway, which is why it lives outside ``agent_hardener.openshell``.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

from agent_hardener.errors import SandboxProvisioningError
from agent_hardener.openshell.lifecycle import (
    CommandRunner,
    OpenShellCommandResult,
    SubprocessCommandRunner,
    combine_results,
    wait_for_http_health,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import BackendServiceSpec

#: Bound on ``docker compose version``: a wedged Docker CLI must not hang the run before it starts.
COMPOSE_PROBE_TIMEOUT_S = 15.0


def _docker_compose_works() -> bool:
    """Return True if the ``docker compose`` (v2 plugin) subcommand is actually available."""
    try:
        result = subprocess.run(
            ["docker", "compose", "version"],
            capture_output=True,
            check=False,
            timeout=COMPOSE_PROBE_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_compose_command() -> list[str]:
    """Return a working Docker Compose invocation.

    Prefers the ``docker compose`` v2 plugin, but only if it actually works; otherwise falls back to
    the standalone ``docker-compose`` v1 binary (common on Homebrew/Colima setups). Raises
    :class:`~agent_hardener.errors.SandboxProvisioningError` when neither works.
    """
    if shutil.which("docker") and _docker_compose_works():
        return ["docker", "compose"]
    if shutil.which("docker-compose"):
        return ["docker-compose"]
    raise SandboxProvisioningError(
        "Docker Compose not found: neither `docker compose` nor `docker-compose` works on this host.",
        remediation="Install the Docker Compose plugin (or the standalone docker-compose), then retry.",
    )


def compose_file_path(spec: BackendServiceSpec, cwd: Path) -> Path:
    """The backend's compose file; a relative ``compose_file`` is taken from ``cwd``."""
    compose_file = spec.compose_file
    if compose_file is None:
        raise ValueError(f"backend '{spec.name}' has no compose_file")
    return compose_file if compose_file.is_absolute() else cwd / compose_file


class BackendManager:
    """Bring NAT-victim backend docker-compose stacks up and down on the host."""

    def __init__(
        self,
        runner: CommandRunner | None = None,
        compose_command: list[str] | None = None,
    ) -> None:
        self.runner = runner or SubprocessCommandRunner()
        self._compose_command = compose_command

    @property
    def compose_command(self) -> list[str]:
        # Resolved on first use: specs without a compose_file never need Compose installed.
        if self._compose_command is None:
            self._compose_command = resolve_compose_command()
        return self._compose_command

    def _compose_args(self, spec: BackendServiceSpec, cwd: Path) -> tuple[list[str], Path]:
        compose_path = compose_file_path(spec, cwd)
        args = [*self.compose_command, "-f", str(compose_path)]
        if spec.compose_project:
            args += ["-p", spec.compose_project]
        return args, compose_path.parent

    def _run(self, args: list[str], cwd: Path) -> OpenShellCommandResult:
        return OpenShellCommandResult.from_completed(args, self.runner.run(args, cwd=cwd))

    def up(self, specs: list[BackendServiceSpec], cwd: Path) -> OpenShellCommandResult:
        """Start each backend stack with a compose_file and wait for its health endpoint."""
        results: list[OpenShellCommandResult] = []
        for spec in specs:
            if spec.compose_file is None:
                continue
            base, project_dir = self._compose_args(spec, cwd)
            result = self._run([*base, "up", "-d", "--build"], project_dir)
            results.append(result)
            if not result.ok:
                return combine_results(results, ok=False)
            if spec.health_url:
                try:
                    wait_for_http_health(spec.health_url, timeout_seconds=120.0)
                except TimeoutError as exc:
                    results.append(OpenShellCommandResult(ok=False, command=base, output=str(exc)))
                    return combine_results(results, ok=False)
        return combine_results(results) if results else OpenShellCommandResult(ok=True, command=[], output="")

    def down(self, specs: list[BackendServiceSpec], cwd: Path) -> OpenShellCommandResult:
        """Tear down each backend stack (best effort across all specs)."""
        results: list[OpenShellCommandResult] = []
        for spec in specs:
            if spec.compose_file is None:
                continue
            base, project_dir = self._compose_args(spec, cwd)
            results.append(self._run([*base, "down", "-v"], project_dir))
        return combine_results(results) if results else OpenShellCommandResult(ok=True, command=[], output="")
