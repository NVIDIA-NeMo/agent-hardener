# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``up``/``down``/``status``/``stage`` commands: the OpenShell sandbox lifecycle."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from agent_hardener.cli._app import (
    ALLOW_CONTEXT_DRIFT_OPTION,
    DEFAULT_MANIFEST,
    ENV_FILE_OPTION,
    VICTIM_IMAGE_OPTION,
    app,
    load_cli_env_file,
)
from agent_hardener.cli._errors import cli_error_boundary
from agent_hardener.display.console import emit, failure_excerpt, ok, status_factory, warn
from agent_hardener.errors import SandboxProvisioningError, VictimBuildError
from agent_hardener.openshell._forward import port_in_use_error

if TYPE_CHECKING:
    from agent_hardener.openshell.config import OpenShellCommandResult

logger = logging.getLogger(__name__)


# Lines of a failed command's output shown on the console; the full output is in the log file.
def _report(
    result: OpenShellCommandResult, *, done: str | None, show_output: bool = False, elapsed: float | None = None
) -> None:
    """Report a lifecycle result as one outcome line; raise a classified error when it failed (else exit 0).

    The raw OpenShell/docker output goes to the log file (and the console with ``--debug``). The console
    gets the ``warning:`` lines it carries, the ``done`` line on success, or the output's error lines on a
    failure (:func:`failure_excerpt`). ``show_output`` prints all of it (``status``, whose output is the
    answer). A lifecycle failure surfaces as ``ok=False`` data, not an exception — raise
    :class:`SandboxProvisioningError` so the surrounding :func:`cli_error_boundary` serializes it for the
    caller (rather than exiting with a bare non-zero code).
    """
    if result.output:
        logger.info("openshell output:\n%s", result.output)
    lines = result.output.strip().splitlines()
    for line in lines:
        if line.startswith("warning:"):
            warn(line.removeprefix("warning:").strip(), markup=False)
    if show_output:
        emit(result.output)
    if not result.ok:
        if not show_output and lines:
            emit(failure_excerpt(result.output))
        raise port_in_use_error(result.output) or SandboxProvisioningError(
            "sandbox lifecycle command failed; see the output above"
        )
    if done:
        ok(done, elapsed=elapsed, markup=False)
    raise typer.Exit(0)


def _simple_lifecycle(config: Path, verb: str) -> None:
    """Run a no-build lifecycle verb (``down``/``status``) against the configured sandbox."""
    from agent_hardener.config import load_config
    from agent_hardener.openshell.lifecycle import OpenShellLifecycle
    from agent_hardener.tools.openshell import openshell_config

    with cli_error_boundary():
        sandbox_config = openshell_config(load_config(str(config)).victim_control)
        lifecycle = OpenShellLifecycle(sandbox_config)
        if verb == "status":
            _report(lifecycle.status(), done=None, show_output=True)
        started = time.monotonic()
        with status_factory(f"Tearing down sandbox {sandbox_config.sandbox}"):
            result = lifecycle.down()
        _report(result, done=f"Sandbox {sandbox_config.sandbox} torn down", elapsed=time.monotonic() - started)


@app.command()
def up(
    config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c"),
    env_file: Path | None = ENV_FILE_OPTION,
    victim_image: str | None = VICTIM_IMAGE_OPTION,
    allow_context_drift: bool = ALLOW_CONTEXT_DRIFT_OPTION,
) -> None:
    """Build and start the OpenShell sandbox for the configured agent."""
    from agent_hardener.config import load_config
    from agent_hardener.openshell.lifecycle import OpenShellLifecycle
    from agent_hardener.tools.openshell import (
        configure_local_docker_host,
        openshell_config,
        prepare_relay_victim,
        with_victim_image,
    )

    with cli_error_boundary():
        configure_local_docker_host()
        load_cli_env_file(env_file)
        victim_control = with_victim_image(
            load_config(str(config)).victim_control, victim_image, allow_context_drift=allow_context_drift
        )
        config_obj = prepare_relay_victim(openshell_config(victim_control))
        lifecycle = OpenShellLifecycle(config_obj)
        started = time.monotonic()
        with status_factory("Building and starting sandbox"):
            result = lifecycle.up()
        elapsed = time.monotonic() - started
        drift = lifecycle.image_context_drift() if result.ok else None
        if drift:
            warn(drift, markup=False)
        _report(result, done=f"Sandbox {config_obj.sandbox} is ready", elapsed=elapsed)


@app.command()
def down(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Tear down the OpenShell sandbox."""
    _simple_lifecycle(config, "down")


@app.command()
def status(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Show OpenShell sandbox status."""
    _simple_lifecycle(config, "status")


@app.command()
def stage(config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c")) -> None:
    """Write the victim's staged build context and print its directory.

    Build and push that directory with any image builder, then pass the image to ``run --victim-image``.
    """
    from agent_hardener.config import load_config
    from agent_hardener.tools.openshell import openshell_config, prepare_relay_victim

    with cli_error_boundary():
        prepared = prepare_relay_victim(openshell_config(load_config(str(config)).victim_control))
        if prepared.relay_victim is None or prepared.build_context is None:
            raise VictimBuildError(
                "the config declares no victim build to stage",
                remediation="Point --config at an agent-hardener.yaml manifest with an agent.dockerfile.",
            )
        typer.echo(str(prepared.build_context.parent))
