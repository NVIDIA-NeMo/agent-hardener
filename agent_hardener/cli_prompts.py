# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive prompting and scaffold writers for ``agent-hardener init``.

Everything here talks to the user (typer prompts, the ``?``-help system) or writes a scaffold
file. The pure discovery/heuristics live in :mod:`agent_hardener.project` and the pure manifest
builders in :mod:`agent_hardener.manifest`; this module orchestrates them into the interactive flow.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import typer
import yaml  # type: ignore[import-untyped]

from agent_hardener.display.console import fail, info, is_rich, ok, path_completion
from agent_hardener.manifest import build_backend
from agent_hardener.project import (
    discover_dockerfiles,
    discover_egress_entries,
    discover_env_files,
    find_project_root,
)

if TYPE_CHECKING:
    from collections.abc import Callable

# Detailed, on-demand explanations shown when the user answers an init prompt with "?". Keeping the
# detail here lets the prompt labels stay short.
_HELP = {
    "project_dir": (
        "The folder holding your agent. init searches it (walking up to the installable root\n"
        "with pyproject.toml/uv.lock) for Dockerfiles. Point at the project root\n"
        "or an agent subfolder; '.' is the current directory."
    ),
    "start_command": (
        "The command that launches your agent inside the sandbox, e.g.\n"
        "`/app/.venv/bin/python -m myagent.serve --port 8000`.\n"
        "Agent Hardener does not generate one: the image and its entrypoint are yours, and it\n"
        "hardens the agent you actually ship rather than a variant of it.\n"
        "Spell out the interpreter in full: OpenShell replaces the image's PATH with its own,\n"
        "so a bare `python` is the system one, not your venv's."
    ),
    "dockerfile": (
        "Path to the Dockerfile Agent Hardener builds for the victim image (relative to the project\n"
        "dir), e.g. deploy/agent/Dockerfile.\n"
        "\n"
        "Your image must satisfy four requirements. Agent Hardener injects the last one for you; the\n"
        "other three are yours, and a missing one fails at sandbox start rather than at build:\n"
        "  1) a 'sandbox' user and group, e.g.\n"
        "       RUN groupadd --system sandbox \\\n"
        "           && useradd --system --gid sandbox --create-home \\\n"
        "               --home-dir /home/sandbox --shell /bin/sh sandbox\n"
        "     Missing -> the container crash-loops with 'sandbox user not found in image'.\n"
        "  2) iproute2 (the `ip` binary), e.g. apt-get install -y --no-install-recommends iproute2\n"
        "     Missing -> 'Network namespace creation failed ... trusted ip helper not found'.\n"
        "  3) `nat` on the DEFAULT PATH (e.g. /usr/local/bin). Setting ENV PATH is NOT enough:\n"
        "     OpenShell replaces the image's PATH with its own. If you install into a venv,\n"
        "     symlink it: RUN ln -s /app/.venv/bin/nat /usr/local/bin/nat\n"
        "     Missing -> \"env: 'nat': No such file or directory\".\n"
        "  4) the aiohttp egress shim on PYTHONPATH - appended to your Dockerfile automatically."
    ),
    "binaries": (
        "Glob patterns for executables/libraries kept in the sandboxed image (comma-separated),\n"
        "e.g. /usr/local/bin/python*,/app/.venv/bin/**."
    ),
    "agent_name": (
        "A label for this agent. Its sandbox, run artifacts, and benign-profile cache are keyed on\n"
        "it, so distinct agents/variants should use different names."
    ),
    "port": (
        "The victim agent serves HTTP on this port inside the sandbox; garak attacks\n"
        "http://127.0.0.1:<port>/v1/chat/completions. Match your workflow's serve port (default 8000)."
    ),
    "secrets_file": (
        "A dotenv file holding the agent's secret values (API keys, tokens) — separate from Iron\n"
        "Swarm's own --env-file. init seeds the secret NAMES from it and offers to fill missing ones."
    ),
    "egress": (
        "Extra external hosts the agent must reach, comma-separated (host or host:port; 443 if the\n"
        "port is omitted). The sandbox denies all other egress. Local/Docker services go in\n"
        "'backends', not here."
    ),
    "backends": (
        "A backend is a host-side service your agent's tools call that runs on your machine — e.g. a\n"
        "Postgres database, a Redis cache, or a local REST API. The sandbox can't reach your host by\n"
        "default, so Agent Hardener opens a route to it (and can start it for you). For each one you give:\n"
        "  • name        — a label for the service (e.g. 'postgres'), used in the sandbox policy\n"
        "  • compose file — a docker-compose to start it (blank if it's already running)\n"
        "  • ports       — the host port(s) the agent calls (allow-listed to host.docker.internal)\n"
        "  • health URL  — optional URL polled to confirm it's up\n"
        "Answer no if your agent doesn't call any host service."
    ),
    "backend_name": (
        "Just a label for this service in the manifest and logs — it doesn't have to match anything\n"
        "in your compose file and doesn't affect routing. Press Enter to accept the default (taken\n"
        "from the compose folder, else backend1/backend2…)."
    ),
    "backend_compose": (
        "Path to a docker-compose file Agent Hardener runs to start/stop this service. Leave blank if\n"
        "the service is already running — then init only opens the sandbox→host route to it."
    ),
    "backend_ports": (
        "The host port(s) the agent calls, comma-separated, e.g. 5432. Each is allow-listed from\n"
        "the sandbox to host.docker.internal:<port>."
    ),
    "backend_health": (
        "Optional URL polled to confirm the backend is up before attacking, e.g.\n"
        "http://127.0.0.1:5432/health. Blank to skip."
    ),
}


def prompt_with_help(
    label: str, help_key: str, *, parse: Callable[[str], object] | None = None, **kwargs: object
) -> object:
    """``typer.prompt`` that prints a detailed explanation when the user types ``?``, then re-asks.

    Appends a dim ``(? for help)`` hint to ``label`` and shows ``_HELP[help_key]`` on ``?``. With
    ``parse`` (e.g. ``int``) the entered value is converted, re-prompting on failure. Keeping the long
    explanation in :data:`_HELP` lets the prompt labels stay short.
    """
    hint = typer.style("(? for help)", fg="bright_black") if is_rich() else "(? for help)"
    full = f"{label} {hint}"
    while True:
        value = typer.prompt(full, **kwargs)
        if isinstance(value, str) and value.strip() == "?":
            _show_help(help_key)
            continue
        if parse is None:
            return value
        try:
            return parse(value)
        except (ValueError, TypeError):
            fail("Please enter a valid value.")


def _show_help(help_key: str) -> None:
    """Print the indented, dim ``_HELP`` blurb for ``help_key`` (used by the ``?`` prompt handlers)."""
    info("")
    for line in _HELP[help_key].split("\n"):
        info(f"  [dim]{line}[/dim]")
    info("")


def confirm_with_help(question: str, help_key: str, *, default: bool = False) -> bool:
    """A yes/no prompt that also accepts ``?`` to print ``_HELP[help_key]``, then re-asks.

    Like :func:`typer.confirm` but with on-demand help — used where a plain confirm left users unsure
    what they were agreeing to (e.g. the host-backend question).
    """
    hint = typer.style("(y/n, ? for help)", fg="bright_black") if is_rich() else "(y/n, ? for help)"
    while True:
        answer = (
            str(typer.prompt(f"{question} {hint}", default="y" if default else "n", show_default=False)).strip().lower()
        )
        if answer == "?":
            _show_help(help_key)
            continue
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        if not answer:
            return default
        fail("Please answer y or n (or ? for help).")


def select_workflow(matches: list[Path], project_root: Path, *, interactive: bool) -> str | None:
    """Resolve the chosen workflow path (relative to ``project_root``) from discovered matches.

    With one match it is used directly. With several, an interactive run prompts a numbered picker;
    a non-interactive run raises (the caller must pass ``--workflow``). With none, an interactive run
    prompts for a path and a non-interactive run returns ``None``.
    """

    def _rel(path: Path) -> str:
        return os.path.relpath(path.resolve(), project_root.resolve())

    if len(matches) == 1:
        return _rel(matches[0])
    if len(matches) > 1:
        if not interactive:
            listed = ", ".join(_rel(match) for match in matches)
            fail(f"multiple workflows found ({listed}); pass --workflow to choose one. Aborted.")
            raise typer.Exit(1)
        info("Multiple workflows found:")
        for index, match in enumerate(matches, start=1):
            info(f"  {index}. {_rel(match)}")
        choice = typer.prompt("Select workflow number", type=int, default=1)
        chosen = matches[min(max(choice, 1), len(matches)) - 1]
        return _rel(chosen)
    if interactive:
        with path_completion(project_root):  # Tab-complete workflow paths relative to the project root
            return (
                str(prompt_with_help("Workflow path within the project root", "workflow", default="")).strip() or None
            )
    return None


def write_garak_scaffold(port: int, *, force: bool = False, path: Path = Path("garak-scan.yaml")) -> Path:
    """Scaffold the editable garak agent_breaker config, returning its path.

    The attacker overlays the resolved target uri onto this at run time, so it is a user-editable
    starting point. An existing file is kept (it may hold model edits) unless ``force``.
    """
    from agent_hardener.agents.attackers.agent_breaker.config import build_agent_breaker_config  # noqa: PLC0415

    if path.exists() and not force:
        info(f"{path} exists; keeping it.")
        return path
    scaffold = build_agent_breaker_config(
        target_uri=f"http://127.0.0.1:{port}/v1/chat/completions",
        report_dir=".agent-hardener/garak_runs",
        report_prefix="agent-breaker",
    )
    path.write_text(yaml.safe_dump(scaffold, sort_keys=False), encoding="utf-8")
    ok(f"Wrote {path}")
    return path


def prompt_backends() -> list[dict]:
    """Interactively collect host-backend definitions (the host DB/API services the agent's tools call).

    The compose file is optional — leaving it blank declares an already-running service, where
    Agent Hardener just opens the sandbox→host route (and health-checks it) without managing the
    container.
    """
    info("")
    info("Backends are host-side services your agent's tools call (a DB, cache, or local API).")
    if not confirm_with_help("Does your agent's tools call a host backend?", "backends", default=False):
        return []
    backends: list[dict] = []
    while True:
        # Ask for the compose file first so the name can default to its folder (the name is only a label).
        with path_completion():  # compose file path is cwd-relative
            compose_file = (
                str(
                    prompt_with_help(
                        "  compose file path (blank if already running)",
                        "backend_compose",
                        default="",
                        show_default=False,
                    )
                ).strip()
                or None
            )
        default_name = (Path(compose_file).parent.name if compose_file else "") or f"backend{len(backends) + 1}"
        name = str(prompt_with_help("  backend name", "backend_name", default=default_name)).strip() or default_name
        ports_raw = str(
            prompt_with_help("  port(s) the agent calls, comma-separated", "backend_ports", default="")
        ).strip()
        ports = [int(part) for part in ports_raw.split(",") if part.strip().isdigit()]
        health_url = (
            str(
                prompt_with_help("  health URL (blank to skip)", "backend_health", default="", show_default=False)
            ).strip()
            or None
        )
        backends.append(build_backend(name, compose_file, ports, health_url))
        if not typer.confirm("Add another backend?", default=False):
            return backends


def resolve_project_layout(pointed_at: Path, *, interactive: bool) -> tuple[str, Path, list[Path], list[Path]]:
    """Resolve the project layout from where the user pointed.

    Walks up to the installable project root so an agent subfolder still yields the correct root,
    then recursively discovers ``Dockerfile`` files under where they pointed.
    Returns ``(project_dir relative to cwd, project_root, dockerfiles)``.
    """
    project_root = find_project_root(pointed_at) or pointed_at
    if interactive and find_project_root(pointed_at) is None:
        info(f"Note: no pyproject.toml/uv.lock found above {pointed_at} — using it as-is.")

    if interactive:
        info(f"Searching {pointed_at} for Dockerfile files…")
    dockerfiles = discover_dockerfiles(pointed_at)
    if interactive and not dockerfiles:
        info("None found — you'll enter a Dockerfile path by hand next.")

    try:
        project_dir_value = os.path.relpath(project_root.resolve(), Path.cwd())
    except ValueError:  # different drive (Windows) — fall back to absolute.
        project_dir_value = str(project_root.resolve())
    return project_dir_value, project_root, dockerfiles


def prompt_byo(detected_dockerfile: str | None, project_root: Path) -> dict:
    """Interactively collect BYO (bring-your-own-image) build details.

    Collects the Dockerfile (defaulting to a detected one, with Tab path-completion relative to the
    project root) and the binary globs kept in the sandboxed image.

    The start command is prompted for separately by :func:`prompt_start_command`; it is not part of
    the image choice. (Historical note: ``manifest._start_command`` used to generate one that served the
    workflow uploaded to ``WORKFLOW_UPLOAD_DEST``, which is what lets the defenders' hardening take
    effect when the victim restarts between rounds. A hand-written command typically serves its own
    baked config and silently discards that, so it stays an advanced YAML-only key.
    """
    info("")
    info("BYO mode: Agent Hardener builds your Dockerfile and runs the image inside the sandbox.")
    info("Your image provides the environment; Agent Hardener still serves and hardens the workflow.")
    info("Whatever your start command needs must be on the image's DEFAULT PATH.")
    with path_completion(project_root):
        kwargs = {"default": detected_dockerfile} if detected_dockerfile else {}
        dockerfile = str(prompt_with_help("  Dockerfile path", "dockerfile", **kwargs)).strip()
    binaries_raw = str(
        prompt_with_help(
            "  binaries to keep in the image (comma-separated globs)",
            "binaries",
            default="/usr/local/bin/python*,/app/.venv/bin/**",
        )
    ).strip()
    binaries = [part.strip() for part in binaries_raw.split(",") if part.strip()]
    return {"dockerfile": dockerfile, "binaries": binaries}


def prompt_launch_mode(dockerfiles: list[Path], project_root: Path) -> dict:
    """Prompt for the Dockerfile that builds the victim image.

    There is no "build it for me" branch any more: Agent Hardener hardens the agent the user ships, so
    building a different image would undercut the result.
    """
    detected = os.path.relpath(dockerfiles[0].resolve(), project_root.resolve()) if dockerfiles else None
    extra = f" (+{len(dockerfiles) - 1} more)" if len(dockerfiles) > 1 else ""
    info("")
    if detected:
        info(f"Victim image [green](detected: {detected}{extra})[/green]")
    else:
        info("Victim image — Agent Hardener runs your Dockerfile; it does not build one for you.")
    return prompt_byo(detected, project_root)


def prompt_start_command(*, yes: bool) -> str:
    """Ask how the agent is launched inside the sandbox.

    Required, with no default worth guessing: the command belongs to the user's image. Under NAT
    Agent Hardener generated one and had to warn when a hand-written command ignored the mutable config
    path; guardrails now arrive as Relay plugin config the agent reads on restart, so the launch
    command and the hardening no longer have to agree about anything.
    """
    if yes:
        msg = "no --start-command given; Agent Hardener cannot guess how your agent is launched"
        raise ValueError(msg)
    return str(prompt_with_help("Start command", "start_command", default="")).strip()


def prompt_egress(project_root: Path) -> list[str]:
    """Show the discovered egress hosts, let the user approve them, and add any that were missed.

    The sandbox is default-deny egress, so every external host the agent calls must be allow-listed.
    Returns the approved + manually-added entries for the manifest's ``egress`` field.
    """
    entries = discover_egress_entries(project_root)
    info("")
    info("The sandbox blocks all outbound network by default — the agent reaches only hosts you allow.")
    info("(Local/Docker host services are configured separately as backends below, not here.)")
    if entries:
        info("Discovered these egress hosts (from the project's code and config):")
        for entry in entries:
            info(f"  - [green]{entry}[/green]")
        if not typer.confirm("Allow these hosts?", default=True):
            entries = []
    else:
        info("No egress hosts found by scanning.")
    extra = str(
        prompt_with_help(
            "Add more egress hosts (comma-separated, blank for none)", "egress", default="", show_default=False
        )
    ).strip()
    for raw in extra.split(","):
        host = raw.strip()
        if host and host not in entries:
            entries.append(host)
    return entries


def resolve_launch_mode(
    dockerfiles: list[Path],
    project_root: Path,
    *,
    yes: bool,
    dockerfile: str | None = None,
    binaries: list[str] | None = None,
) -> dict:
    """Decide which Dockerfile builds the victim image, as a ``{dockerfile, binaries}`` mapping.

    There is no longer a second mode. Agent Hardener hardens the agent the user ships, so it does not
    build a generic image for them — the image is always theirs, and ``binaries`` scopes egress
    because we cannot infer the venv layout of an image we did not write.
    """
    if dockerfile:
        return {"dockerfile": dockerfile, "binaries": list(binaries or [])}
    if yes:
        if not dockerfiles:
            msg = "no Dockerfile found under the project; pass --dockerfile (Agent Hardener does not build one for you)"
            raise ValueError(msg)
        if not binaries:
            # Never defaulted: these globs scope which processes may egress, and a guessed default
            # silently over-permits a layout we cannot see inside.
            msg = "a detected Dockerfile still needs --binary globs, e.g. --binary '/app/.venv/bin/**'"
            raise ValueError(msg)
        # relpath on both *resolved* paths, not Path.relative_to: a symlinked root (macOS /tmp ->
        # /private/tmp) makes the discovered path and the project root disagree, and relative_to
        # raises rather than resolving it.
        chosen = os.path.relpath(dockerfiles[0].resolve(), project_root.resolve())
        return {"dockerfile": chosen, "binaries": list(binaries)}
    return prompt_launch_mode(dockerfiles, project_root)


def resolve_pointed_at(project_dir: Path | None, *, yes: bool) -> Path:
    """Resolve where init looks for the agent project, returning the path the user pointed at.

    Explicit ``--project-dir`` wins; ``--yes`` uses the cwd; otherwise prompt (cwd-relative, Tab-
    completed) with ``.`` (the current directory) as the default.
    """
    if project_dir is not None:
        return project_dir
    if yes:
        return Path()
    with path_completion():  # Tab-complete the project path relative to the cwd
        entered = str(prompt_with_help("Path to your agent project", "project_dir", default=".")).strip()
    return Path(entered) if entered else Path()


def resolve_agent_name(name: str | None, default_name: str, *, yes: bool) -> str:
    """Resolve the agent name for ``init``: explicit ``--name`` wins, ``--yes`` takes the default, else prompt."""
    if name:
        return name
    if yes:
        return default_name
    return str(prompt_with_help("Agent name", "agent_name", default=default_name)).strip() or default_name


def resolve_secrets_file(secrets_file: str | None, pointed_at: Path, project_root: Path, *, yes: bool) -> str:
    """Resolve the agent's dotenv path for ``init`` (separate from Agent Hardener's ``--env-file``).

    Explicit ``--secrets-file`` wins; ``--yes`` uses ``.env``; otherwise offer a ``.env`` detected in the
    agent's own project root (not Agent Hardener's cwd), labelled ``(detected)``. The returned path is
    relative to cwd so the manifest resolves it (e.g. ``../agents-lab/.env``).
    """
    if secrets_file:
        return secrets_file.strip()
    if yes:
        return ".env"
    # Look in the agent's project root (and the exact dir the user pointed at), never Agent Hardener's cwd.
    detected_env = discover_env_files(project_root, pointed_at)
    env_default = os.path.relpath(detected_env[0], Path.cwd()) if detected_env else ".env"
    # typer.prompt labels aren't Rich-rendered, so style the "(detected)" marker with typer.style (ANSI);
    # gate on is_rich() so a non-TTY / NO_COLOR run doesn't leak raw escape codes.
    marker = typer.style("(detected)", fg="green") if is_rich() else "(detected)"
    label = f"Agent secrets file {marker}" if detected_env else "Agent secrets file"
    with path_completion():  # the secrets-file path is cwd-relative, so complete against the cwd
        return str(prompt_with_help(label, "secrets_file", default=env_default)).strip()
