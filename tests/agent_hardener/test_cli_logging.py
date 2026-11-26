# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the CLI prints, and what it keeps for the log file: console levels, ``--debug``, the log path."""

from __future__ import annotations

import io
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

from agent_hardener import cli
from agent_hardener.cli._app import app
from agent_hardener.cli._errors import cli_error_boundary
from agent_hardener.display import console, progress
from agent_hardener.display.console import failure_excerpt
from agent_hardener.display.progress import ConsoleRenderer
from agent_hardener.errors import AgentHardenerError
from agent_hardener.loggers import FILE_ONLY, configure_cli_logging
from agent_hardener.openshell.lifecycle import OpenShellCommandResult

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def out(monkeypatch: pytest.MonkeyPatch) -> io.StringIO:
    """Route the shared console to a plain (non-terminal) buffer."""
    buffer = io.StringIO()
    con = Console(file=buffer, force_terminal=False, width=80)
    monkeypatch.setattr(console, "get_console", lambda: con)
    monkeypatch.setattr(progress, "get_console", lambda: con)
    return buffer


def _log_lines(path: Path) -> str:
    for handler in logging.getLogger("agent_hardener").handlers:
        handler.flush()
    return path.read_text(encoding="utf-8")


# --- console vocabulary ----------------------------------------------------------------------------------


def test_levels_render_with_their_glyph_and_long_lines_are_not_wrapped(out: io.StringIO) -> None:
    long_path = "/very/long/" + "segment/" * 20 + "file.log"
    console.say("ok", "Sandbox ready", elapsed=12.42)
    console.say("warn", "image drifted")
    console.say("hint", f"log: {long_path}")
    console.say("section", "Cleanup")
    console.say("detail", "hidden without --debug")
    console.say("info", "[not markup]")

    assert out.getvalue().splitlines() == [
        "✓ Sandbox ready (12.4s)",
        "! image drifted",
        f"  → log: {long_path}",  # one line even past the 80-column width
        "",
        "── Cleanup ──",
        "[not markup]",
    ]


def test_detail_prints_only_with_debug(out: io.StringIO) -> None:
    console.set_debug(True)
    console.detail("12 relay events")
    assert out.getvalue() == "  · 12 relay events\n"


@pytest.mark.parametrize(("seconds", "shown"), [(0.42, "0.4s"), (185, "3m05s"), (3725, "1h02m")])
def test_format_elapsed(seconds: float, shown: str) -> None:
    assert console.format_elapsed(seconds) == shown


# --- log records on the console, and in the file ---------------------------------------------------------


def test_default_console_shows_warnings_as_one_line_and_the_file_keeps_everything(
    out: io.StringIO, tmp_path: Path
) -> None:
    path = configure_cli_logging("run", debug=False, logs_dir=tmp_path)
    log = logging.getLogger("agent_hardener.session")
    log.debug("polling tick")
    log.info("openshell command completed")
    try:
        raise ValueError("boom")
    except ValueError:
        log.warning("could not collect attack traces\nsecond line", exc_info=True)
    log.error("defenders manager failed critically")
    log.error("report blob", extra=FILE_ONLY)
    logging.getLogger("httpx").warning("third-party noise")

    assert out.getvalue().splitlines() == [
        "! could not collect attack traces (details in log)",
        "✗ defenders manager failed critically",
    ]
    text = _log_lines(path)
    for expected in ("polling tick", "openshell command completed", "Traceback", "report blob", "third-party noise"):
        assert expected in text


def test_debug_console_prints_every_record_in_the_log_format(out: io.StringIO, tmp_path: Path) -> None:
    configure_cli_logging("run", debug=True, logs_dir=tmp_path)
    logging.getLogger("agent_hardener.session").debug("polling tick")
    logging.getLogger("agent_hardener.events").info("agent_progress echo")  # rendered by the event renderer
    logging.getLogger("agent_hardener.run").info("final report", extra=FILE_ONLY)
    logging.getLogger("httpx").info("HTTP Request: POST")

    lines = out.getvalue().splitlines()
    assert len(lines) == 2
    assert lines[0].endswith("DEBUG   session: polling tick")
    assert lines[1].endswith("INFO    httpx: HTTP Request: POST")


def test_the_log_file_is_created_only_when_something_is_logged(tmp_path: Path) -> None:
    path = configure_cli_logging("status", debug=False, logs_dir=tmp_path / "logs")
    assert not path.parent.exists()
    logging.getLogger("agent_hardener").debug("first record")
    assert path.is_file()
    assert path.name.startswith("status-")


def test_the_log_file_can_be_named_by_the_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = tmp_path / "job" / "agent-hardener.log"
    monkeypatch.setenv("AGENT_HARDENER_LOG_FILE", str(chosen))

    path = configure_cli_logging("run", debug=False, logs_dir=tmp_path / "ignored")
    logging.getLogger("agent_hardener").info("kept for the upload")

    assert path == chosen
    assert "kept for the upload" in chosen.read_text(encoding="utf-8")


def test_error_boundary_keeps_the_traceback_in_the_log_not_the_console(out: io.StringIO, tmp_path: Path) -> None:
    path = configure_cli_logging("run", debug=False, logs_dir=tmp_path)

    with pytest.raises(typer.Exit), cli_error_boundary():
        raise AgentHardenerError("sandbox did not start", remediation="Check Docker, then retry.")

    assert out.getvalue().splitlines() == ["✗ sandbox did not start", "  → Check Docker, then retry."]
    assert "Traceback" in _log_lines(path)


# --- the CLI: --debug anywhere, and the log path at the end ----------------------------------------------


def test_debug_after_the_command_name_is_moved_in_front(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr(cli, "app", lambda *, args, **_kw: seen.append(args))
    cli.main(["run", "--rounds", "2", "--debug", "--", "--debug"])
    assert seen == [["--debug", "run", "--rounds", "2", "--", "--debug"]]


def test_every_command_ends_with_its_log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["--debug", "run", "--config", "missing.yaml"])

    assert result.exit_code == 1
    last = result.output.strip().splitlines()[-1]
    assert last.startswith("  → log: ")
    assert (tmp_path / last.removeprefix("  → log: ")).is_file()


# --- the live renderer off a terminal ---------------------------------------------------------------------


def test_off_a_terminal_the_run_keeps_its_phases_agents_and_summaries(out: io.StringIO) -> None:
    render = ConsoleRenderer()
    render("phase_started", {"phase": "attackers", "count": 1})
    render("status_started", {"label": "spinner only"})
    render("agent_failed", {"agent_id": "a", "agent_name": "garak", "error": "TimeoutError: no answer\ntrace"})
    render("agent_completed", {"agent_id": "b", "agent_name": "replay", "ok": True, "duration_seconds": 3.0})
    render(
        "attack_summary",
        {"attacks": [{"agent_name": "garak", "records": [{}, {}], "metadata": {"attempts": 40}}]},
    )

    assert out.getvalue().splitlines() == [
        "",
        "── ATTACK (1 agent) ──",
        "✗ garak: TimeoutError: no answer",
        "✓ replay (3.0s)",
        "✓ ATTACK complete — 2 / 40 attempts landed",
    ]


def test_an_attack_that_made_no_attempts_is_a_warning_not_a_pass(out: io.StringIO) -> None:
    ConsoleRenderer()(
        "attack_summary", {"attacks": [{"agent_name": "garak", "records": [], "metadata": {"attempts": 0}}]}
    )

    assert out.getvalue().splitlines() == ["! ATTACK complete — no attempts were made; see the attacker log"]


def test_phase_headlines_count_runs_and_show_a_repeated_failure_once(out: io.StringIO) -> None:
    render = ConsoleRenderer()
    no_trace = {"agent_name": "guardrails", "ok": False, "error": "no execution trace"}
    render("phase_started", {"phase": "defenders", "count": 2})
    for _ in range(3):
        render("agent_completed", {**no_trace, "phase": "defenders"})
    render("agent_completed", {"agent_name": "policy", "ok": True, "phase": "defenders", "duration_seconds": 2.0})
    render("phase_completed", {"phase": "defenders", "count": 0, "ok": False})
    render("phase_started", {"phase": "validators", "count": 2})
    replay = {"agent_name": "replay", "ok": False, "summary": "blocked 2/5 hits", "phase": "validators"}
    render("agent_completed", replay)
    render("agent_completed", {"agent_name": "benign", "ok": True, "summary": "9/9 complied", "phase": "validators"})
    render("phase_completed", {"phase": "validators", "count": 2, "ok": False})
    render("phase_started", {"phase": "validators", "count": 0})
    render("phase_completed", {"phase": "validators", "count": 0, "ok": True})
    render("phase_started", {"phase": "defenders", "count": 1})
    render("phase_completed", {"phase": "defenders", "count": 1, "ok": True})

    assert out.getvalue().splitlines() == [
        "",
        "── DEFENSE (2 agents) ──",
        "✗ guardrails: no execution trace",
        "✓ policy (2.0s)",
        "✗ DEFENSE complete — 1 / 4 defender run(s) succeeded; 2 repeated result(s) not shown again (all are in the log)",
        "",
        "── VALIDATION (2 agents) ──",
        "✗ replay: blocked 2/5 hits",
        "✓ benign: 9/9 complied",
        "✗ VALIDATION complete — 1 / 2 validator(s) succeeded",
        "",
        "── VALIDATION (0 agents) ──",
        "! VALIDATION complete — no validator ran",
        "",
        "── DEFENSE (1 agent) ──",
        "✓ DEFENSE complete — 1 defender run(s)",
    ]


def _ok_result(output: str, *, ok: bool = True) -> OpenShellCommandResult:
    return OpenShellCommandResult(ok=ok, command=["openshell"], output=output)


def test_up_failure_shows_the_warnings_and_the_tail_of_the_output(monkeypatch: pytest.MonkeyPatch) -> None:
    blob = "warning: provider key not refreshed\n" + "\n".join(f"build step {i}" for i in range(30))

    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def up(self) -> OpenShellCommandResult:
            return _ok_result(blob, ok=False)

    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda *_a, **_k: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.openshell_config", lambda _vc: SimpleNamespace(sandbox="sb"))
    monkeypatch.setattr("agent_hardener.tools.openshell.prepare_relay_victim", lambda cfg: cfg)
    monkeypatch.setattr("agent_hardener.tools.openshell.with_victim_image", lambda vc, *_a, **_kw: vc)
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p: SimpleNamespace(victim_control="vc"))
    monkeypatch.setattr("agent_hardener.openshell.lifecycle.OpenShellLifecycle", FakeLifecycle)

    result = CliRunner().invoke(app, ["up"])

    assert result.exit_code == 1
    assert "! provider key not refreshed" in result.output
    assert "build step 29" in result.output
    assert "build step 14" not in result.output  # only the last 15 lines; the log has the rest
    assert "✗ sandbox lifecycle command failed; see the output above" in result.output


def test_up_names_the_taken_port_instead_of_blaming_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLifecycle:
        def __init__(self, _cfg: object) -> None:
            pass

        def up(self) -> OpenShellCommandResult:
            return _ok_result("Gateway Info\nError: Port 8000 is already in use by another process.", ok=False)

    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda *_a, **_k: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.openshell_config", lambda _vc: SimpleNamespace(sandbox="sb"))
    monkeypatch.setattr("agent_hardener.tools.openshell.prepare_relay_victim", lambda cfg: cfg)
    monkeypatch.setattr("agent_hardener.tools.openshell.with_victim_image", lambda vc, *_a, **_kw: vc)
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p: SimpleNamespace(victim_control="vc"))
    monkeypatch.setattr("agent_hardener.openshell.lifecycle.OpenShellLifecycle", FakeLifecycle)

    result = CliRunner().invoke(app, ["up"])

    assert result.exit_code == 1
    assert "✗ Port 8000 is already in use by another process, so the victim cannot be reached on it" in result.output
    assert "lsof -i :8000" in result.output
    assert "Check the Docker daemon" not in result.output


@pytest.mark.parametrize(("ready", "line"), [(True, "✓ Reusing the Ready sandbox"), (False, "! No Ready sandbox")])
def test_run_reuse_says_whether_the_sandbox_is_reused(monkeypatch: pytest.MonkeyPatch, ready: bool, line: str) -> None:
    session = SimpleNamespace(
        victim_control="vc", benign_suite_path=None, benign_validators=[], model_copy=lambda **_kw: session
    )
    monkeypatch.setattr("agent_hardener.tools.openshell.configure_local_docker_host", lambda: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.load_local_env", lambda *_a, **_k: None)
    monkeypatch.setattr("agent_hardener.tools.openshell.openshell_config", lambda _vc: "cfg")
    monkeypatch.setattr("agent_hardener.tools.openshell.sandbox_is_ready", lambda _c: ready)
    monkeypatch.setattr("agent_hardener.preflight.run.preflight_run", lambda _session, **_options: None)
    monkeypatch.setattr("agent_hardener.config.load_config", lambda _p, **_kw: session)
    monkeypatch.setattr("agent_hardener.runtime.runner.run_mission", lambda *_a, **_kw: SimpleNamespace(success=True))

    result = CliRunner().invoke(app, ["run", "--reuse"])

    assert result.exit_code == 0
    assert line in result.output


def test_output_events_render_by_level(out: io.StringIO) -> None:
    render = ConsoleRenderer()
    render("output", {"line": "✓ Sandbox ready (1.0s)", "message": "Sandbox ready", "level": "ok", "elapsed": 1.0})
    render("output", {"line": "docker: no space left on device", "level": "output"})
    render("output", {"line": "legacy line without a level"})
    assert out.getvalue().splitlines() == [
        "✓ Sandbox ready (1.0s)",
        "docker: no space left on device",
        "legacy line without a level",
    ]


def test_summaries_say_why_an_agent_failed(out: io.StringIO) -> None:
    render = ConsoleRenderer()
    render(
        "attack_summary", {"attacks": [{"agent_name": "garak", "ok": False, "error": "TimeoutError", "records": []}]}
    )
    render("validator_summary", {"validators": [{"agent_name": "replay", "ok": False, "error": "judge down\nmore"}]})
    assert out.getvalue().splitlines() == [
        "✗ ATTACK complete — 0 hits",
        "  garak: TimeoutError",
        "  replay: failed: judge down",
    ]


def test_on_a_terminal_a_failed_agent_row_carries_its_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    con = Console(file=io.StringIO(), force_terminal=True, width=100)
    monkeypatch.setattr(console, "get_console", lambda: con)
    monkeypatch.setattr(progress, "get_console", lambda: con)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_PLAIN", raising=False)
    render = ConsoleRenderer()
    render("phase_started", {"phase": "defenders", "count": 1})
    render("agent_started", {"agent_id": "d", "agent_name": "openshell-defender"})
    render("agent_failed", {"agent_id": "d", "agent_name": "openshell-defender", "error": "x" * 400})

    assert render._progress is not None
    (task,) = render._progress.tasks
    assert task.description.startswith("[red]✗ openshell-defender: xxx")
    assert task.description.endswith("…[/red]")  # the reason is cut; the log has all of it
    render("phase_completed", {"phase": "defenders"})


def test_a_long_warning_is_cut_and_a_broken_console_never_raises(
    out: io.StringIO, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configure_cli_logging("run", debug=False, logs_dir=tmp_path)
    logging.getLogger("agent_hardener").warning("y" * 500)
    assert out.getvalue() == "! " + "y" * 299 + "…\n"

    monkeypatch.setattr(logging, "raiseExceptions", False)
    monkeypatch.setattr(console, "warn", lambda *_a, **_kw: (_ for _ in ()).throw(OSError("closed")))
    logging.getLogger("agent_hardener").warning("still fine")  # handleError swallows it


def test_no_log_line_when_nothing_was_logged_and_absolute_outside_the_cwd(out: io.StringIO, tmp_path: Path) -> None:
    from agent_hardener.cli._app import _print_log_path  # noqa: PLC0415

    _print_log_path(tmp_path / "never-written.log")
    written = tmp_path / "run.log"
    written.write_text("x", encoding="utf-8")
    _print_log_path(written)
    assert out.getvalue() == f"  → log: {written}\n"


def test_failure_excerpt_leads_with_the_error_lines() -> None:
    launcher = "victim start command: " + "x" * 400
    output = "\n".join(
        [
            *(f"plugin {i}" for i in range(20)),
            "Error:   \u00d7 code: 'Some requested entity was not found', message: \"sandbox not found\"",
            "  │ (an existence check, before the create)",
            launcher,
            "timed out waiting for victim health at http://127.0.0.1:8017/health",
            "Error:   \u00d7 Port 8000 is already in use by another process.",
            "  │ COMMAND PID",
            "  │   kill <PID>",
            "",
            "no configured provider credentials found in .env",
        ]
    )

    shown = failure_excerpt(output).splitlines()

    assert shown == [
        "… 23 other line(s) in the log",
        "timed out waiting for victim health at http://127.0.0.1:8017/health",
        "Error:   \u00d7 Port 8000 is already in use by another process.",
        "  │ COMMAND PID",
        "  │   kill <PID>",
        "no configured provider credentials found in .env",
    ]


def test_failure_excerpt_falls_back_to_the_tail_and_cuts_long_lines() -> None:
    output = "\n".join([*(f"step {i}" for i in range(20)), "y" * 400])

    shown = failure_excerpt(output, limit=3).splitlines()

    assert shown == ["… 18 other line(s) in the log", "step 18", "step 19", f"{'y' * 300}…"]
    assert failure_excerpt("") == ""


def test_failure_excerpt_keeps_the_last_line_even_when_it_is_not_an_error() -> None:
    shown = failure_excerpt("step 1\nError: build failed\nstep 2\nsee the diagnostics above").splitlines()

    assert shown == ["… 2 other line(s) in the log", "Error: build failed", "see the diagnostics above"]
