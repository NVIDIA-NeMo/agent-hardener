# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The ``run`` command: the full attack/defend/validate cycle against the configured agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
import typer
from typer.core import TyperCommand

from agent_hardener.cli._app import (
    ALLOW_CONTEXT_DRIFT_OPTION,
    DEFAULT_ENV_FILE,
    DEFAULT_MANIFEST,
    VICTIM_IMAGE_OPTION,
    app,
)
from agent_hardener.display.console import fail, info, ok

if TYPE_CHECKING:
    from agent_hardener.models import SessionConfig

# Sentinel value `--replay` takes when passed bare (no path): replay the latest run's hitlog.
REPLAY_LATEST = "__latest__"
# Sentinel value `--with-traces` takes when passed bare: load the traces recorded next to the hitlog.
TRACES_SAVED = "__saved__"


class ReplayCommand(TyperCommand):
    """Make ``--replay`` and ``--with-traces`` accept an optional value.

    Bare ``--replay`` replays the latest hitlog; ``--replay PATH`` replays that specific file. Bare
    ``--with-traces`` loads the traces recorded next to that hitlog; ``--with-traces PATH`` loads that
    file. Typer
    builds every option as either a flag or a value-taking option and drops Click's ``flag_value``, so
    we finish wiring the optional-value behaviour on the built Click option here (mirrors what
    ``click.Option(is_flag=False, flag_value=...)`` sets).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        sentinels = {"replay": REPLAY_LATEST, "with_traces": TRACES_SAVED}
        for param in self.params:
            if isinstance(param, click.Option) and param.name in sentinels:
                param.is_flag = False
                param.flag_value = sentinels[param.name]
                param._flag_needs_value = True


def _apply_initial_overrides(relay_plugins: Path | None, policy: Path | None) -> None:
    """Validate and announce the optional ``run --relay-plugins/--policy`` initial overrides."""
    for label, path in (("relay-plugins", relay_plugins), ("policy", policy)):
        if path is None:
            continue
        if not path.is_file():
            fail(f"--{label} file not found: {path}")
            raise typer.Exit(1)
        info(f"Initial {label}: {path}")


def _resolve_replay(replay: str | None, session_config: SessionConfig, with_traces: str | None = None) -> SessionConfig:
    """Fold the three-way ``--replay`` value into the config: drop attackers, add a preloaded suite.

    ``None`` = flag absent (no change); ``REPLAY_LATEST`` = bare flag → newest hitlog on disk;
    anything else = an explicit hitlog path. ``with_traces`` (see :func:`_resolve_replay_traces`)
    attaches the attack traces recorded by the run that produced the hitlog.
    """
    from agent_hardener.models import PreloadedAttackConfig
    from agent_hardener.storage import find_latest_hitlog
    from agent_hardener.swarm_tracker import run_logs_root

    if replay is None:
        if with_traces is not None:
            fail("--with-traces needs --replay: a live run collects its traces from the trace provider.")
            raise typer.Exit(1)
        return session_config

    if replay == REPLAY_LATEST:
        garak_settings = session_config.garak
        explicit_dir = garak_settings.report_dir if garak_settings else None
        # Attackers write hitlogs run-scoped under <root_dir>/run-logs/<run_id>/round_<N>/garak/ unless
        # garak.report_dir is explicitly set (which the attacker then writes to directly). Mirror that here.
        search_dir = Path(explicit_dir) if explicit_dir else run_logs_root(session_config.storage.root_dir)
        hitlog_path = find_latest_hitlog(search_dir)
        if hitlog_path is None:
            fail(f"No hitlog found under {search_dir}. Run without --replay first to generate attack hits.")
            raise typer.Exit(1)
        preload_name = "latest-run"
    else:
        hitlog_path = Path(replay)
        if not hitlog_path.is_file():
            fail(f"Replay hitlog not found: {hitlog_path}")
            raise typer.Exit(1)
        preload_name = "uploaded"

    info(f"Replaying attacks from: {hitlog_path}")
    traces_path = _resolve_replay_traces(with_traces, hitlog_path)
    return session_config.model_copy(
        update={
            "attackers": [],
            "preloaded_attacks": [
                PreloadedAttackConfig(
                    name=preload_name,
                    path=hitlog_path.resolve(),
                    source="garak-agent-breaker",
                    traces_path=traces_path,
                )
            ],
        }
    )


def _resolve_replay_traces(with_traces: str | None, hitlog_path: Path) -> Path | None:
    """The validated attack-traces file for a replay with traces, or ``None`` to replay without them.

    ``TRACES_SAVED`` (bare flag) = the ``attack_traces.json`` recorded in the hitlog's round
    directory; anything else = an explicit file. A missing or invalid file fails fast: silently
    replaying without traces would report results that do not reflect them.
    """
    from agent_hardener.traces.attacks import ATTACK_TRACES_FILENAME, load_recording

    if with_traces is None:
        return None
    if with_traces == TRACES_SAVED:
        # Hitlogs are written under round_<N>/garak/, the recording under round_<N>/.
        path = hitlog_path.resolve().parent.parent / ATTACK_TRACES_FILENAME
    else:
        path = Path(with_traces).resolve()
    if not path.is_file():
        fail(f"Attack traces not found: {path}. Record them with traces.provider on, or pass --with-traces PATH.")
        raise typer.Exit(1)
    try:
        count = len(load_recording(path))
    except ValueError as exc:
        fail(f"Not a valid attack traces file: {path} ({exc.__class__.__name__})")
        raise typer.Exit(1) from exc
    info(f"Replaying with {count} recorded attack trace(s) from: {path}")
    return path


def _apply_benign_suite(benign_suite: Path | None, session_config: SessionConfig) -> SessionConfig:
    """Point the run at a supplied benign-suite CSV, or leave the config untouched when none is given."""
    if benign_suite is None:
        return session_config
    if not benign_suite.is_file():
        fail(f"Benign suite not found: {benign_suite}")
        raise typer.Exit(1)
    info(f"Using supplied benign suite: {benign_suite}")
    return session_config.model_copy(update={"benign_suite_path": benign_suite.resolve()})


@app.command(cls=ReplayCommand)
def run(
    config: Path = typer.Option(DEFAULT_MANIFEST, "--config", "-c", help="Manifest or full session config."),
    env_file: Path = typer.Option(DEFAULT_ENV_FILE, "--env-file", help="Dotenv file with secrets."),
    rounds: int = typer.Option(1, "--rounds", help="Number of iterative hardening rounds."),
    mission_id: str | None = typer.Option(None, "--mission-id", help="Mission id prefix."),
    reuse: bool = typer.Option(
        False, "--reuse", help="Reuse the running sandbox if it's already Ready (skip rebuild)."
    ),
    no_cleanup: bool = typer.Option(False, "--no-cleanup", help="Leave sandbox + backends running."),
    replay: str | None = typer.Option(
        None,
        "--replay",
        metavar="[PATH]",
        help="Skip the attacker and replay recorded garak hits. Bare replays the latest run's hitlog; "
        "pass a path to a garak hitlog (.jsonl) to replay that specific file.",
    ),
    with_traces: str | None = typer.Option(
        None,
        "--with-traces",
        metavar="[PATH]",
        help="With --replay: also load the victim's attack traces. Bare loads the attack_traces.json recorded "
        "next to the hitlog; pass a path to load that file instead.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show the full final report (transcripts, Garak detail, policy diffs) in a pager. "
        "Raw build/infra output always goes to agent-hardener.log, not inline.",
    ),
    benign_suite: Path | None = typer.Option(
        None,
        "--benign-suite",
        help="Benign-suite CSV (tool,payload,label,rationale,persona) to validate against. Generate one with "
        "`agent-hardener synth-benign`. When omitted, benign validation is skipped for this run.",
    ),
    relay_plugins: Path | None = typer.Option(
        None,
        "--relay-plugins",
        help="Initial NeMo Relay plugins.toml to run from, overriding the manifest's seed. Defenders still "
        "harden on top. New outbound endpoints it introduces are not auto-discovered — declare them in "
        "the manifest's egress/backends.",
    ),
    policy: Path | None = typer.Option(
        None,
        "--policy",
        help="Initial OpenShell policy YAML to run from, overriding the default permissive policy. Backend "
        "and discovered-egress rules are still layered on top; defenders still harden it further.",
    ),
    victim_image: str | None = VICTIM_IMAGE_OPTION,
    allow_context_drift: bool = ALLOW_CONTEXT_DRIFT_OPTION,
) -> None:
    """Run the full attack/defend/validate cycle against the configured agent.

    ``run`` consumes a benign suite; it does not generate one. Supply a suite with ``--benign-suite`` (or
    a manifest ``benign_suite_path``); generate suites separately with ``agent-hardener synth-benign``.
    """
    from agent_hardener.cli._errors import cli_error_boundary
    from agent_hardener.config import load_config
    from agent_hardener.display.progress import build_console_renderer
    from agent_hardener.model_check import preflight_configured_models
    from agent_hardener.runtime.runner import run_mission
    from agent_hardener.tools.openshell import (
        configure_local_docker_host,
        load_local_env,
        openshell_config,
        sandbox_is_ready,
        with_victim_image,
    )

    # One boundary over the whole command: any classified failure (bad manifest, sandbox down, victim
    # unreachable, …) is serialized to $AGENT_HARDENER_ERROR_FILE for the caller and printed cleanly here.
    with cli_error_boundary():
        configure_local_docker_host()
        load_local_env(env_file)
        # Fail fast on a mistyped/unreachable operator-overridden model before building the sandbox.
        preflight_configured_models()
        _apply_initial_overrides(relay_plugins, policy)
        session_config = load_config(str(config), relay_plugins=relay_plugins, policy=policy)
        session_config = _resolve_replay(replay, session_config, with_traces)
        session_config = _apply_benign_suite(benign_suite, session_config)
        if victim_image is not None:
            session_config = session_config.model_copy(
                update={
                    "victim_control": with_victim_image(
                        session_config.victim_control, victim_image, allow_context_drift=allow_context_drift
                    )
                }
            )

        # --reuse resolves to the skip_initial_up primitive: skip the rebuild only if a Ready sandbox
        # is already running, otherwise build fresh.
        skip_initial_up = reuse and sandbox_is_ready(openshell_config(session_config.victim_control))
        if reuse:
            (ok if skip_initial_up else info)(
                "Reusing Ready sandbox; skipping rebuild." if skip_initial_up else "No Ready sandbox; building fresh."
            )

        # One subscriber renders all one-way presentation (output lines, spinners, phase banners); the
        # runner emits everything as events. MissionError/VictimUnavailableError propagate to the boundary.
        result = run_mission(
            session_config,
            rounds=rounds,
            verbose=verbose,
            mission_id=mission_id,
            skip_initial_up=skip_initial_up,
            no_cleanup=no_cleanup,
            on_event=build_console_renderer(verbose=verbose),
        )
        raise typer.Exit(0 if result.success else 1)
