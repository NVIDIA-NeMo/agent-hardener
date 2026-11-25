# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``setup`` command: provision one-time prerequisites (garak venv + OpenShell gateway)."""

from __future__ import annotations

import time

import typer

from agent_hardener.cli._app import app
from agent_hardener.cli._errors import cli_error_boundary
from agent_hardener.display.console import ok, section, status_factory, warn
from agent_hardener.openshell.gateway import register_local_gateway


@app.command()
def setup(
    force: bool = typer.Option(False, "--force", "-f", help="Recreate the garak venv even if present."),
) -> None:
    """Provision one-time prerequisites: the dedicated garak venv and the local OpenShell gateway.

    garak installs into its own venv (default ``~/.agent-hardener/garak-venv``; override with
    ``$AGENT_HARDENER_GARAK_PYTHON``) because it pulls ``litellm`` (``httpx>=0.28``) and ``torch`` that
    would otherwise conflict with agent-hardener's own dependencies. Idempotent unless ``--force``.

    The gateway step is best-effort: a not-connected gateway warns but does not fail setup.
    Installing the OpenShell CLI/service itself is out of scope (needs brew/sudo) — see `just setup`.
    """
    from agent_hardener.garak_venv import provision_garak_venv

    section("Provisioning garak venv")
    started = time.monotonic()
    with cli_error_boundary(), status_factory("Installing garak"):
        python = provision_garak_venv(force=force)
    ok(f"garak ready: {python}", elapsed=time.monotonic() - started, markup=False)

    section("Configuring OpenShell gateway")
    result = register_local_gateway()
    (ok if result.ok else warn)(result.message, markup=False)
