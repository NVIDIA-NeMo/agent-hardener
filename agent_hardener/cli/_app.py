# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared Typer app and CLI-wide defaults.

Command modules import ``app`` from here and register themselves with ``@app.command(...)``; the package
``__init__`` imports those modules so registration happens on import. Kept separate from ``__init__`` to
avoid a circular import (``__init__`` → command modules → ``app``).
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

import typer

# The LLM/agent stack emits many cosmetic third-party warnings (Pydantic serializer warnings,
# ChatNVIDIA structured-output notices, deprecations). They're noise for CLI users, so silence
# all warnings for every command. (The synth/replay paths re-apply this inside a context manager
# too, since some dependencies reset warnings.filters mid-run.)
warnings.filterwarnings("ignore")

app = typer.Typer(
    help="Agent Hardener — run a security war-game against your Relay-connected agent.",
    no_args_is_help=True,
    add_completion=False,  # hide the built-in --install-completion/--show-completion options
)

# Commands whose stdout is machine-readable output (a path, JSON): no trailing log-path line on them.
_MACHINE_OUTPUT_COMMANDS = frozenset({"stage", "inspect"})


@app.callback()
def _configure(
    ctx: typer.Context,
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Print every log record to the console — engine detail, OpenShell commands, tracebacks and "
        "third-party library logs. The log file always has them. Also: AGENT_HARDENER_LOG_LEVEL=debug.",
    ),
) -> None:
    """Wire logging for this invocation and print where its log file is when the command ends."""
    from agent_hardener import __version__
    from agent_hardener.loggers import configure_cli_logging

    command = ctx.invoked_subcommand or "agent-hardener"
    debug = debug or os.environ.get("AGENT_HARDENER_LOG_LEVEL", "").strip().lower() == "debug"
    log_path = configure_cli_logging(command, debug=debug)
    logging.getLogger("agent_hardener.cli").debug("agent-hardener %s %s (debug=%s)", __version__, command, debug)
    if command not in _MACHINE_OUTPUT_COMMANDS:
        ctx.call_on_close(lambda: _print_log_path(log_path))


def _print_log_path(path: Path) -> None:
    """Close every command with the location of its full diagnostic log (when anything was logged)."""
    from agent_hardener.display.console import hint

    if not path.is_file():
        return
    try:
        shown = path.relative_to(Path.cwd())
    except ValueError:
        shown = path
    hint(f"log: {shown}", markup=False)


DEFAULT_MANIFEST = Path("agent-hardener.yaml")
DEFAULT_ENV_FILE = Path(".env")


ENV_FILE_OPTION = typer.Option(None, "--env-file", help=f"Dotenv file with secrets [default: {DEFAULT_ENV_FILE}].")


def load_cli_env_file(env_file: Path | None) -> Path:
    """Load ``--env-file`` (or the default); only a file the operator named must exist."""
    from agent_hardener.tools.openshell import load_local_env

    path = env_file or DEFAULT_ENV_FILE
    load_local_env(path, explicit=env_file is not None)
    return path


def _victim_image(value: str | None) -> str | None:
    if value is None:
        return None
    image = value.strip()
    if not image:
        raise typer.BadParameter("name an image reference, e.g. registry.example.com/agent:1")
    from agent_hardener.display.console import info

    info(f"Using pre-built victim image: {image}")
    return image


VICTIM_IMAGE_OPTION = typer.Option(
    None,
    "--victim-image",
    callback=_victim_image,
    help=(
        "Pre-built victim image to run instead of building locally with docker. Build it from the context "
        "`agent-hardener stage` writes."
    ),
)

ALLOW_CONTEXT_DRIFT_OPTION = typer.Option(
    False,
    "--allow-context-drift",
    help="Run a --victim-image that was not built from the context `agent-hardener stage` writes now, "
    "with a warning instead of failing.",
)
