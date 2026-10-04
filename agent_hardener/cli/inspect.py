# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``inspect`` command: detect an agent project's layout, secrets, and egress (offline, read-only)."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_hardener.cli._app import app
from agent_hardener.display.console import info


@app.command()
def inspect(
    project_dir: Path = typer.Option(Path(), "--project-dir", help="Path to your agent project."),
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable JSON (no banner)."),
) -> None:
    """Detect an agent project's image, secrets, and egress (non-interactive, offline).

    The read-only counterpart to ``init``'s detection phase: it never writes or prompts. Studio shells
    this to pre-fill its upload wizard, then feeds the confirmed answers back to ``init --yes``.
    """
    import json as json_lib
    import os

    from agent_hardener.cli_prompts import resolve_project_layout
    from agent_hardener.project import (
        default_agent_name,
        discover_backend_ports,
        discover_egress_entries,
        discover_env_files,
        secret_names_from_file,
    )

    project_dir_value, project_root, dockerfiles = resolve_project_layout(project_dir, interactive=False)

    def rel_to_root(path: Path) -> str:
        return os.path.relpath(path.resolve(), project_root.resolve())

    dockerfile_list = [rel_to_root(dockerfile) for dockerfile in dockerfiles]

    env_files = discover_env_files(project_root, project_dir)
    secrets_file = rel_to_root(env_files[0]) if env_files else ""
    secret_names = secret_names_from_file(env_files[0]) if env_files else []

    result = {
        "project_dir": project_dir_value,
        "dockerfiles": dockerfile_list,
        "default_agent_name": default_agent_name(project_dir_value, None),
        "default_port": 8000,
        "secrets_file": secrets_file,
        "secret_names": secret_names,
        "egress": discover_egress_entries(project_root),
        "backend_ports": discover_backend_ports(project_root),
    }

    if as_json:
        typer.echo(json_lib.dumps(result))
        return

    info("Detected agent project:")
    info(f"  dockerfiles   {', '.join(dockerfile_list) or '(none)'}")
    info(f"  agent name    {result['default_agent_name']}")
    info(f"  secrets file  {secrets_file or '(none detected)'}")
    info(f"  secret names  {', '.join(secret_names) or '(none)'}")
    info(f"  egress        {', '.join(result['egress']) or '(none)'}")
    info(f"  backend ports {', '.join(map(str, result['backend_ports'])) or '(none)'}")
