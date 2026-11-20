# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``agent-hardener`` command-line interface.

A single entrypoint for running Agent Hardener against a user's agent. ``run`` is the bash script
replacement: expand the manifest, bring up host backends, build + harden the agent in OpenShell, and
clean up (the garak agent_breaker attacker spawns the garak CLI itself). ``init`` scaffolds a manifest
plus a ``garak-scan.yaml`` config; ``up``/``down``/``status`` expose the sandbox lifecycle, and ``stage`` writes
the victim build context for hosts that build the image elsewhere and pass it back with ``--victim-image``.

One module per command lives beside this file; importing them here registers each on the shared ``app``.
The import order fixes the order commands appear in ``--help``.
"""

from __future__ import annotations

import sys

from agent_hardener.cli._app import app
from agent_hardener.cli import run, synth_benign, serve, lifecycle, setup, init, inspect, skills

__all__ = ["app", "init", "inspect", "lifecycle", "main", "run", "serve", "setup", "skills", "synth_benign"]


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point: run the app, accepting ``--debug`` after the command name too.

    ``--debug`` is an app-level option (it configures logging before any command runs), so Typer only
    accepts it before the command. ``agent-hardener run --debug`` is the natural spelling, so it is moved
    to the front here. Anything after a ``--`` separator is left alone.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    end = args.index("--") if "--" in args else len(args)
    if "--debug" in args[:end]:
        args.remove("--debug")
        args.insert(0, "--debug")
    app(args=args, prog_name="agent-hardener")
