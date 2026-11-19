# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``init`` command: scaffold an ``agent-hardener.yaml`` manifest (auto-detecting the NAT project)."""

from __future__ import annotations

from pathlib import Path

import typer

from agent_hardener.cli._app import DEFAULT_MANIFEST, app
from agent_hardener.display.console import banner, fail, info, ok, section


def _parse_backend_flag(spec: str) -> dict:
    """Parse a ``--backend NAME:PORT[,PORT2]`` flag into a route-only ``backends`` entry."""
    from agent_hardener.manifest import build_backend

    name, _, ports_raw = spec.partition(":")
    name = name.strip()
    ports = [int(part) for part in ports_raw.split(",") if part.strip().isdigit()]
    if not name or not ports:
        raise typer.BadParameter(f"--backend must be NAME:PORT[,PORT2]; got {spec!r}")
    return build_backend(name, None, ports, None)


def _check_byo_flags(dockerfile: str | None, binary: list[str]) -> None:
    """Reject ``--dockerfile`` without ``--binary`` before any prompting or scaffolding.

    ``RelayVictimSpec`` refuses a BYO image with no binary globs — they scope which processes may
    egress, which cannot be inferred from an unknown image layout. Catching it here turns a
    validation error raised minutes into a run into an actionable message with the fix in it.
    """
    if dockerfile and not binary:
        fail("--dockerfile requires at least one --binary glob, e.g. --binary '/app/.venv/bin/**'. Aborted.")
        raise typer.Exit(1)


@app.command()
def init(
    project_dir: Path | None = typer.Option(None, "--project-dir", help="Path to your NAT project."),
    name: str | None = typer.Option(None, "--name", help="Agent name."),
    dockerfile: str | None = typer.Option(
        None,
        "--dockerfile",
        help="Dockerfile that builds your agent's image (path within the project). Agent Hardener hardens "
        "the agent you ship, so it does not build one for you. Requires --binary. Your image must carry "
        "a 'sandbox' user/group, iproute2, and whatever the start command needs on the default PATH "
        "(setting ENV PATH is not enough) — answer '?' at the interactive prompt for the full contract.",
    ),
    start_command: str | None = typer.Option(
        None, "--start-command", help="Command that launches your agent inside the sandbox."
    ),
    binary: list[str] = typer.Option(
        [],
        "--binary",
        help="Glob of binaries kept in the sandboxed BYO image (repeatable). Required with --dockerfile.",
    ),
    port: int | None = typer.Option(None, "--port", help="Victim service port (default 8000)."),
    secrets: str | None = typer.Option(None, "--secrets", help="Comma-separated secret names (overrides the file)."),
    secrets_file: str | None = typer.Option(None, "--secrets-file", help="Dotenv file holding the agent's secrets."),
    egress: list[str] = typer.Option(
        [], "--egress", help="Allow-listed egress host[:port] the agent reaches (repeatable)."
    ),
    backend: list[str] = typer.Option(
        [],
        "--backend",
        help="Route-only host backend the agent's tools call, as NAME:PORT[,PORT2] (repeatable). "
        "Rewrites the agent's localhost:PORT to host.docker.internal:PORT and opens the sandbox->host route.",
    ),
    output: Path = typer.Option(DEFAULT_MANIFEST, "--output", "-o", help="Manifest path to write."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive: accept defaults, no prompts."),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite an existing manifest."),
) -> None:
    """Scaffold an ``agent-hardener.yaml`` manifest (auto-detecting your agent project)."""
    from agent_hardener.cli_prompts import (
        prompt_backends,
        prompt_egress,
        prompt_start_command,
        prompt_with_help,
        resolve_agent_name,
        resolve_launch_mode,
        resolve_pointed_at,
        resolve_project_layout,
        resolve_secrets_file,
        write_garak_scaffold,
    )
    from agent_hardener.manifest import build_manifest, render_manifest_yaml
    from agent_hardener.project import (
        append_env,
        default_agent_name,
        missing_secrets,
        secret_names_from_file,
    )

    # Orient the user first — explain Agent Hardener and what this command produces — before any prompt
    # or the "already exists" check, so they know what they're running.
    if not yes:
        banner(
            "Agent Hardener — security war-game for your agent",
            [
                "It attacks your agent, lets defenders harden it, then replays the attacks to verify.",
                f"This command generates a manifest file named {output}, plus garak-scan.yaml,",
                "and prompts for your agent's secrets.",
                "Bracketed values are defaults (or detected — labelled when so); press Enter to accept.",
            ],
        )

    # Don't silently clobber an existing manifest (and any manual edits in it).
    if output.exists() and not force and (yes or not typer.confirm(f"{output} exists. Overwrite?", default=False)):
        fail(f"{output} already exists; pass --force to overwrite. Aborted.")
        raise typer.Exit(1)

    # Prompt only for scaffold fields that weren't supplied as options. Dim section dividers group the
    # interactive prompts into phases (skipped under --yes, which doesn't prompt).
    def sect(title: str) -> None:
        if not yes:
            section(title)

    sect("Project")
    pointed_at = resolve_pointed_at(project_dir, yes=yes)
    project_dir_value, project_root, dockerfiles = resolve_project_layout(pointed_at, interactive=not yes)
    sect("Launch")
    _check_byo_flags(dockerfile, binary)
    byo = resolve_launch_mode(dockerfiles, project_root, yes=yes, dockerfile=dockerfile, binaries=list(binary))
    start_command_value = start_command or prompt_start_command(yes=yes)

    sect("Agent")
    agent_name = resolve_agent_name(name, default_agent_name(project_dir_value, None), yes=yes)
    if port is None:
        port = 8000 if yes else int(prompt_with_help("Agent port", "port", parse=int, default="8000"))

    sect("Secrets")
    secrets_file_value = resolve_secrets_file(secrets_file, pointed_at, project_root, yes=yes)
    env_path = Path(secrets_file_value)

    # Seed the secret names: explicit --secrets wins; else default to the keys already in the file.
    if secrets is not None:
        secret_names = [s.strip() for s in secrets.split(",") if s.strip()]
    else:
        secret_names = secret_names_from_file(env_path) or ["INFERENCE_API_KEY"]

    if yes:
        # Non-interactive: take egress + route-only backends from the flags.
        egress, backends = list(egress), [_parse_backend_flag(spec) for spec in backend]
    else:
        sect("Network")
        egress = prompt_egress(project_root)
        sect("Backends")
        backends = prompt_backends()

    manifest = build_manifest(
        name=agent_name,
        project_dir=project_dir_value,
        dockerfile=byo["dockerfile"],
        start_command=start_command_value,
        binaries=byo["binaries"],
        port=port,
        secrets=secret_names,
        secrets_file=secrets_file_value,
        backends=backends,
        egress=egress,
    )
    output.write_text(render_manifest_yaml(manifest), encoding="utf-8")
    ok(f"Wrote {output}")
    # Always regenerate the garak config on init so its target port matches the manifest (a kept
    # stale file would attack the wrong port). The user has already opted into (re)scaffolding here.
    garak_path = write_garak_scaffold(port, force=True)

    # Prompt for any listed secret whose value isn't already in the env/file (hidden input).
    pending = missing_secrets(secret_names, env_path)
    collected: dict[str, str] = {}
    if pending and not yes:
        info(f"Enter values for missing secrets (saved to {env_path}; leave blank to skip; input hidden):")
        for secret_name in pending:
            value = typer.prompt(f"  {secret_name}", default="", hide_input=True, show_default=False)
            if value:
                collected[secret_name] = value
    if collected:
        append_env(env_path, collected)
        ok(f"Wrote {len(collected)} secret(s) to {env_path}")

    still_missing = [s for s in pending if s not in collected]
    if still_missing:
        fail(f"Still unset: {', '.join(still_missing)} — fill in {env_path} before running.")

    # Summarize the artifacts and where they landed (absolute paths), then the next command.
    info("")
    info("Generated:")
    info(f"  manifest      {output.resolve()}")
    info("                the war-game config — agents, victim target, and run settings")
    info(f"  garak config  {garak_path.resolve()}")
    info("                editable garak agent_breaker scaffold the attacker overlays each run")
    info(f"  secrets       {env_path.resolve()}{'' if env_path.exists() else ' (create + fill before running)'}")
    info("                env file holding your agent's API keys and credentials")
    info("\nNext:")
    info(f"  uv run agent-hardener synth-benign --config {output}   # generate the benign test suite")
    info(f"  uv run agent-hardener run --config {output}            # run Agent Hardener against your agent")
