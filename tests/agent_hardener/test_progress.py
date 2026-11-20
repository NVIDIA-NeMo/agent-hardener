# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the live console renderer and the event-logger observer hook."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pytest
from rich.console import Console

from agent_hardener.display import progress
from agent_hardener.loggers import JsonlEventLogger

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _reporter(
    monkeypatch: pytest.MonkeyPatch, *, verbose: bool = False
) -> tuple[progress.ConsoleRenderer, io.StringIO]:
    buf = io.StringIO()
    con = Console(file=buf, force_terminal=True, width=90)
    monkeypatch.setattr(progress, "get_console", lambda: con)
    monkeypatch.setattr(progress, "is_rich", lambda: True)  # treat the forced-terminal console as styled
    return progress.ConsoleRenderer(verbose=verbose), buf


def test_phase_banners_and_highlights(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, buf = _reporter(monkeypatch)
    reporter("phase_started", {"phase": "attackers", "count": 1})
    reporter("attack_summary", {"attacks": [{"agent_name": "agent_breaker", "records": [{"probe": "p"}] * 18}]})
    reporter("phase_started", {"phase": "defenders", "count": 2})
    reporter(
        "defender_summary",
        {
            "defenders": [
                {"agent_name": "openshell-policy-defender", "ok": True, "policy_patches": []},
                {"agent_name": "defender-guardrails", "ok": True, "policy_patches": [{}, {}, {}]},
            ],
            "policy_patches": [{}, {}, {}],
        },
    )
    reporter("phase_started", {"phase": "validators", "count": 2})
    reporter("phase_completed", {"phase": "validators", "count": 2, "ok": True})
    out = buf.getvalue()
    assert "ATTACK" in out
    assert "DEFENSE" in out
    assert "VALIDATION" in out
    assert "18 hits" in out
    assert "\x1b[1;31m" in out  # non-zero hits styled bold red
    assert "0 patches" in out
    assert "3 patches" in out
    assert "VALIDATION complete" in out
    assert "Attacker results" not in out  # table is verbose-only


def test_attack_table_only_in_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, buf = _reporter(monkeypatch, verbose=True)
    # No garak_job_id -> no report.jsonl -> transcript skipped; the summary table still shows.
    reporter("attack_summary", {"attacks": [{"agent_name": "agent_breaker", "records": [{"probe": "p"}]}]})
    assert "Attacker results" in buf.getvalue()


def test_attack_transcript_printed_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    from rich.text import Text  # noqa: PLC0415

    import agent_hardener.display.renderers.garak_attacker as ga  # noqa: PLC0415

    monkeypatch.setattr(ga, "attack_transcript_for_attacks", lambda _attacks: Text("TRANSCRIPT-HERE"))
    reporter, buf = _reporter(monkeypatch, verbose=True)
    reporter("attack_summary", {"attacks": [{"agent_name": "agent_breaker", "records": [{"probe": "p"}]}]})
    out = buf.getvalue()
    assert "Attacker results" in out  # summary table
    assert "TRANSCRIPT-HERE" in out  # chat-style transcript printed at stage end


def test_failed_defender_shown_red(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, buf = _reporter(monkeypatch)
    # A non-dict entry is skipped; the failed defender is shown red.
    reporter("defender_summary", {"defenders": ["junk", {"agent_name": "openshell-policy-defender", "ok": False}]})
    out = buf.getvalue()
    assert "failed" in out


def test_defense_reasoning_only_in_verbose(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "defenders": [{"agent_name": "openshell-policy-defender", "ok": True, "policy_patches": []}],
        "policy_patches": [],
    }
    quiet, qbuf = _reporter(monkeypatch)
    quiet("defender_summary", payload)
    assert "openshell-policy-defender" in qbuf.getvalue()
    assert "Policy and guardrail results" not in qbuf.getvalue()  # reasoning is verbose-only

    loud, lbuf = _reporter(monkeypatch, verbose=True)
    loud("defender_summary", payload)
    assert "Policy and guardrail results" in lbuf.getvalue()  # defender reasoning at stage end


def test_unknown_and_internal_events_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    reporter, buf = _reporter(monkeypatch)
    reporter("round_started", {"round_id": "s"})  # no handler
    reporter("phase_started", {"phase": "victim", "count": 1})  # internal phase -> no banner
    reporter("phase_completed", {"phase": "defenders", "count": 2, "ok": True})  # only validators completion prints
    assert buf.getvalue() == ""


def test_build_console_renderer_always_returns_renderer(monkeypatch: pytest.MonkeyPatch) -> None:
    # Always a renderer (so `output` prints even off a TTY); rich features gate internally.
    monkeypatch.setattr(progress, "is_rich", lambda: True)
    assert isinstance(progress.build_console_renderer(verbose=True), progress.ConsoleRenderer)
    monkeypatch.setattr(progress, "is_rich", lambda: False)
    assert isinstance(progress.build_console_renderer(verbose=False), progress.ConsoleRenderer)


def test_output_prints_off_tty_but_banners_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    printed: list[str] = []
    monkeypatch.setattr(progress, "emit", printed.append)
    monkeypatch.setattr(progress, "is_rich", lambda: False)  # not a styled terminal
    renderer = progress.ConsoleRenderer()
    renderer("output", {"line": "hello-line"})
    renderer("phase_started", {"phase": "attackers", "count": 1})  # rich banner suppressed off a TTY
    assert printed == ["hello-line"]


# --- event-logger observer hook -----------------------------------------------------------


def test_event_logger_notifies_observer(tmp_path: Path) -> None:
    seen: list[tuple[str, dict]] = []
    logger = JsonlEventLogger(tmp_path / "events.jsonl", observer=lambda event, payload: seen.append((event, payload)))
    logger.emit("phase_started", phase="attackers", count=1)
    logger.close()
    assert seen == [("phase_started", {"phase": "attackers", "count": 1})]
    assert (tmp_path / "events.jsonl").exists()  # still written to disk


def test_event_logger_observer_error_does_not_propagate(tmp_path: Path) -> None:
    def boom(_event: str, _payload: dict) -> None:
        raise RuntimeError("display blew up")

    logger = JsonlEventLogger(tmp_path / "events.jsonl", observer=boom)
    logger.emit("phase_started", phase="attackers")  # must not raise — a display error can't abort a run
    logger.close()


def test_session_logger_console_level_gated_by_quiet(tmp_path: Path) -> None:
    import logging  # noqa: PLC0415

    from agent_hardener.loggers import build_round_logger  # noqa: PLC0415

    def levels(*, quiet: bool) -> dict[str, int]:
        logger = build_round_logger("s", tmp_path, quiet=quiet)
        return {type(h).__name__: h.level for h in logger.handlers}

    quiet = levels(quiet=True)
    loud = levels(quiet=False)
    assert quiet["StreamHandler"] == logging.WARNING  # console hushed when quiet
    assert loud["StreamHandler"] == logging.INFO  # full INFO to console when verbose
    assert quiet["FileHandler"] == logging.INFO  # the .agent-hardener/round.log file is never hushed
    assert loud["FileHandler"] == logging.INFO


class _FakeStatus:
    """Records spinner lifecycle + label updates for the status-spinner tests."""

    def __init__(self) -> None:
        self.updates: list[str] = []
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")

    def update(self, label: str) -> None:
        self.updates.append(label)


def test_status_spinner_updates_on_synth_phase_and_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    status = _FakeStatus()
    texts: list[str] = []

    def _spinner(text: str) -> _FakeStatus:
        texts.append(text)
        return status

    out = io.StringIO()
    monkeypatch.setattr(progress, "is_rich", lambda: True)
    monkeypatch.setattr(progress, "get_console", lambda: Console(file=out, width=200))
    monkeypatch.setattr(progress, "_Spinner", _spinner)

    renderer = progress.ConsoleRenderer()
    renderer("status_started", {"label": "Synthesizing benign suite", "detail": "reading the description"})
    assert texts == ["Synthesizing benign suite — reading the description…"]
    assert status.events == ["start"]
    renderer(
        "synth_phase",
        {
            "phase": "gap_detector",
            "label": "checked the profile for gaps",
            "detail": "2 gaps unresolved",
            "seconds": 9.2,
            "next": "composing interview questions",
        },
    )
    renderer("other_event", {"label": "ignored"})  # no handler → ignored
    # The driver brackets the interview prompt with these events; the spinner stops then restarts.
    renderer("interview_started", {})
    assert status.events == ["start", "stop"]
    renderer("interview_completed", {})
    renderer("status_completed", {"label": "Synthesizing benign suite"})
    assert out.getvalue() == "✓ checked the profile for gaps — 2 gaps unresolved (9s)\n"
    assert status.updates == ["Synthesizing benign suite — composing interview questions…"]
    # Resumed after the interview, and stopped on completion.
    assert status.events == ["start", "stop", "start", "stop"]


def test_synth_phase_prints_unwrapped_off_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(progress, "is_rich", lambda: False)
    monkeypatch.setattr(progress, "get_console", lambda: Console(file=out, width=40))

    renderer = progress.ConsoleRenderer()
    renderer(
        "synth_phase",
        {"phase": "critic", "label": "filtered and deduplicated requests", "detail": "58 → 42 kept", "seconds": 125},
    )
    renderer("synth_phase", {"phase": "interview_ask", "label": "recorded the interview answers", "seconds": None})

    assert out.getvalue().splitlines() == [
        "✓ filtered and deduplicated requests — 58 → 42 kept (2m 05s)",  # one line though wider than 40 columns
        "✓ recorded the interview answers",
    ]


def test_the_spinner_clock_survives_a_long_label(monkeypatch: pytest.MonkeyPatch) -> None:
    out = io.StringIO()
    monkeypatch.setattr(progress, "get_console", lambda: Console(file=out, width=50, force_terminal=True))

    spinner = progress._Spinner("x" * 200)
    spinner.start()
    spinner.update("y" * 200)
    spinner.stop()

    assert "0:00:00" in out.getvalue()


@pytest.mark.parametrize(("seconds", "text"), [(4.4, "4s"), (59.6, "1m 00s"), (125, "2m 05s")])
def test_durations_switch_to_minutes_past_a_minute(seconds: float, text: str) -> None:
    assert progress.format_duration(seconds) == text


def test_status_events_noop_off_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "is_rich", lambda: False)  # off a styled terminal → no spinner
    renderer = progress.ConsoleRenderer()
    renderer("status_started", {"label": "benign recon"})  # must not raise / create a status
    renderer("interview_started", {})
    renderer("status_completed", {"label": "benign recon"})
