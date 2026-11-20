# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``skills`` commands: install the Agent Hardener skill into a project for coding agents to use."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import typer

from agent_hardener.cli._app import app
from agent_hardener.display.console import info, ok

skills_app = typer.Typer(help="Install the Agent Hardener skill for coding agents.", no_args_is_help=True)
app.add_typer(skills_app, name="skills")


class Target(StrEnum):
    """Skills directories a coding agent reads (keys of ``agent_hardener.skills.TARGETS``)."""

    claude = "claude"
    agents = "agents"


@skills_app.command("install")
def install(
    target: list[Target] = typer.Option(
        [Target.claude],
        "--target",
        "-t",
        help="Where to install (repeatable): 'claude' for .claude/skills/ (Claude Code), "
        "'agents' for .agents/skills/ (Codex, Cursor).",
    ),
    project_dir: Path = typer.Option(Path(), "--project-dir", help="Project to install into (default: current dir)."),
    force: bool = typer.Option(False, "--force", "-f", help="Replace an existing copy (e.g. after upgrading)."),
) -> None:
    """Copy the Agent Hardener skill into the project, so a coding agent knows how to run and read it."""
    from agent_hardener.skills import install_skill

    for name in dict.fromkeys(t.value for t in target):
        skill_dir, written = install_skill(project_dir, target=name, force=force)
        if written:
            ok(f"Installed the agent skill in {skill_dir.resolve()}")
        else:
            info(f"Kept the existing skill in {skill_dir.resolve()}; pass --force to replace it.")
