# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``init`` command: scaffold an ``agent-hardener.yaml`` manifest (auto-detecting your agent project)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, NoReturn

import typer
import yaml  # type: ignore[import-untyped]
from pydantic import ValidationError
from ruamel.yaml import YAML

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
    project_dir: Path | None = typer.Option(None, "--project-dir", help="Path to your agent project."),
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
    force: bool = typer.Option(
        False, "--force", "-f", help="Replace an existing manifest from scratch instead of updating it."
    ),
) -> None:
    """Scaffold an ``agent-hardener.yaml`` manifest (auto-detecting your agent project).

    Run again on an existing manifest, it updates it in place: the project is inspected again, the
    flags you pass are applied, and everything else you set is kept. It prints what changed.
    """
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
    from agent_hardener.project import default_agent_name, secret_names_from_file

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

    # A manifest already here is updated, not clobbered: it holds the user's answers and hand edits.
    # --force is the explicit way to start again from scratch.
    if output.exists() and not force:
        _update_existing(
            output,
            yes=yes,
            project_dir=project_dir,
            name=name,
            dockerfile=dockerfile,
            start_command=start_command,
            binary=list(binary),
            port=port,
            secrets=secrets,
            secrets_file=secrets_file,
            egress=list(egress),
            backend=list(backend),
        )
        return

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
    _write_atomically(output, render_manifest_yaml(manifest))
    ok(f"Wrote {output}")
    # A fresh scaffold starts the garak config over too. (The run sets its target URI either way.)
    garak_path = write_garak_scaffold(port, force=True)
    _finish(output, garak_path, secret_names, env_path, yes=yes)


def _finish(output: Path, garak_path: Path, secret_names: list[str], env_path: Path, *, yes: bool) -> None:
    """Collect missing secret values, then summarize what init produced and what to run next."""
    from agent_hardener.project import append_env, missing_secrets

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


def _write_atomically(path: Path, text: str) -> None:
    """Replace *path* with *text* in one step, so a failure mid-write never leaves a truncated manifest."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _reason(exc: Exception) -> str:
    """A one-line reason for *exc*; pydantic's multi-line report is cut down to each field and its message."""
    if isinstance(exc, ValidationError):
        return "; ".join(f"{'.'.join(str(part) for part in err['loc'])}: {err['msg']}" for err in exc.errors())
    return str(getattr(exc, "message", None) or exc)


def _not_changed(output: Path, reason: str) -> NoReturn:
    """Report a failed update and exit: nothing was written, so the saved manifest is exactly as it was."""
    fail(f"Could not update {output}: {reason}", markup=False)
    info(f"{output} was not changed.")
    raise typer.Exit(1)


def _merged(kept: list[str], found: list[str]) -> list[str]:
    """*kept* followed by whatever *found* adds, in order and without repeats."""
    return list(dict.fromkeys([*kept, *found]))


def _update_existing(output: Path, *, yes: bool, **flags: Any) -> None:
    """Update the manifest already at *output* in place, keeping everything the user set.

    The project is inspected again and the flags given now are applied; every other value in the file is
    kept, including the keys init never writes (``harness``, ``env``, ``garak``, ``defenders``,
    ``overrides``). The file is written once, at the end, only if something changed — so a failure at any
    point leaves it exactly as it was, and an update that changes nothing keeps the user's comments.
    """
    from agent_hardener.cli_prompts import write_garak_scaffold
    from agent_hardener.manifest import AgentManifest

    try:
        text = output.read_text(encoding="utf-8")
        saved = yaml.safe_load(text)
        AgentManifest.model_validate(saved)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        fail(f"{output} exists but cannot be read: {_reason(exc)}", markup=False)
        info(f"{output} was not changed. Fix it, or pass --force to replace it from scratch.")
        raise typer.Exit(1) from exc

    if not yes and not typer.confirm(
        f"{output} exists. Update it? Your settings are kept; only what you pass now or what changed in the "
        "project is replaced",
        default=True,
    ):
        info(f"{output} was not changed. Pass --force to replace it from scratch, or --output for a new file.")
        raise typer.Exit(1)

    try:
        updated = _updated_manifest(saved, yes=yes, **flags)
    except (OSError, ValueError, ValidationError, typer.BadParameter) as exc:
        _not_changed(output, _reason(exc))

    changes = _changes(saved, updated)
    if changes:
        _write_atomically(output, _with_changes(text, updated))
        ok(f"Updated {output}")
        for line in changes:
            info(line)
    else:
        ok(f"{output} is up to date; nothing changed, so the file was left as it was.")
    # Keep the user's garak edits: the run sets the target URI from the manifest anyway.
    garak_path = write_garak_scaffold(updated["agent"]["port"])
    agent = updated["agent"]
    _finish(output, garak_path, agent["secrets"], Path(agent["secrets_file"]), yes=yes)


def _updated_manifest(
    saved: dict[str, Any],
    *,
    yes: bool,
    project_dir: Path | None,
    name: str | None,
    dockerfile: str | None,
    start_command: str | None,
    binary: list[str],
    port: int | None,
    secrets: str | None,
    secrets_file: str | None,
    egress: list[str],
    backend: list[str],
) -> dict[str, Any]:
    """*saved* with the project inspected again and the flags given now applied (pure apart from reads).

    Raises:
        ValueError: The project or its Dockerfile is gone, or a flag is malformed.
        ValidationError: The result is not a valid manifest.
    """
    from agent_hardener.cli_prompts import resolve_project_layout
    from agent_hardener.manifest import AgentManifest
    from agent_hardener.project import discover_egress_entries, secret_names_from_file

    agent = dict(saved["agent"])
    pointed_at = project_dir if project_dir is not None else Path(agent["project_dir"])
    if not pointed_at.is_dir():
        raise ValueError(f"the project directory {str(pointed_at)!r} does not exist; pass --project-dir")
    project_dir_value, project_root, _ = resolve_project_layout(pointed_at, interactive=False)
    dockerfile_value = dockerfile or agent["dockerfile"]
    if not (project_root / dockerfile_value).is_file():
        raise ValueError(f"the Dockerfile {dockerfile_value!r} is not in {project_root}; pass --dockerfile")

    secrets_file_value = secrets_file or agent.get("secrets_file", ".env")
    if secrets is not None:
        secret_names = [s.strip() for s in secrets.split(",") if s.strip()]
    else:
        secret_names = _merged(agent.get("secrets", []), secret_names_from_file(Path(secrets_file_value)))

    egress_value = list(egress) or list(agent.get("egress", []))
    if not egress and not yes:
        # Hosts the project calls now that the saved list does not allow yet; a removed one stays allowed.
        new_hosts = [host for host in discover_egress_entries(project_root) if host not in egress_value]
        if new_hosts:
            info("The project now calls hosts the manifest does not allow:")
            for host in new_hosts:
                info(f"  - [green]{host}[/green]")
            if typer.confirm("Allow these hosts too?", default=True):
                egress_value += new_hosts

    agent.update(
        name=name or agent["name"],
        project_dir=project_dir_value,
        dockerfile=dockerfile_value,
        start_command=start_command or agent["start_command"],
        binaries=list(binary) or agent["binaries"],
        port=port if port is not None else agent.get("port", 8000),
        secrets=secret_names,
        secrets_file=secrets_file_value,
    )
    if egress_value:
        agent["egress"] = egress_value
    else:
        agent.pop("egress", None)
    updated = {**saved, "agent": agent}
    if backend:
        updated["backends"] = [_parse_backend_flag(spec) for spec in backend]
    AgentManifest.model_validate(updated)
    return updated


def _with_changes(text: str, updated: dict[str, Any]) -> str:
    """*text* with its agent block and backends set to *updated*'s, keeping the file's comments and layout.

    Only values that differ are replaced, through ruamel's round-trip mode, so the user's own comments,
    the scaffold's hints and the key order all survive an update.
    """
    rt = YAML()
    rt.preserve_quotes = True
    rt.width = 4096
    doc = rt.load(text)
    agent = doc["agent"]
    for key, value in updated["agent"].items():
        if agent.get(key) != value:
            agent[key] = value
    if (doc.get("backends") or []) != updated.get("backends", []):
        doc["backends"] = updated["backends"]
    out = io.StringIO()
    rt.dump(doc, out)
    return out.getvalue()


def _changes(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """One line per manifest setting an update changed, so it never replaces a value silently."""
    lines: list[str] = []
    old_agent, new_agent = before.get("agent") or {}, after.get("agent") or {}
    for key in [*new_agent, *(k for k in old_agent if k not in new_agent)]:
        old, new = old_agent.get(key), new_agent.get(key)
        if old == new:
            continue
        if isinstance(old, list) or isinstance(new, list):
            added = [item for item in new or [] if item not in (old or [])]
            removed = [item for item in old or [] if item not in (new or [])]
            lines.append(f"  {key:<14} {' '.join([*(f'+{i}' for i in added), *(f'-{i}' for i in removed)])}")
        else:
            lines.append(
                f"  {key:<14} {old if old is not None else '(none)'} -> {new if new is not None else '(none)'}"
            )
    old_names = [b.get("name") for b in before.get("backends") or []]
    new_names = [b.get("name") for b in after.get("backends") or []]
    if (before.get("backends") or []) != (after.get("backends") or []):
        lines.append(
            f"  {'backends':<14} {', '.join(map(str, old_names)) or '(none)'} -> "
            f"{', '.join(map(str, new_names)) or '(none)'}"
        )
    return lines
