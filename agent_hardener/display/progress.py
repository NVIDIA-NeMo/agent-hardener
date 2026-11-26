# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live console renderer: turns run events into colored, phase-structured console output.

Wired by the CLI as the sole ``on_event`` subscriber. All one-way presentation flows through here as
events: ``output`` highlight lines, ``status_started``/``status_completed`` spinners with an elapsed clock
(sandbox bring-up, health, benign synthesis), ATTACK / DEFENSE / VALIDATION banners, live per-agent progress,
and benign synthesis steps (each ``synth_phase`` prints a line with what the step produced and moves the
spinner to the step running now; the spinner is paused around the interview via
``interview_started``/``interview_completed`` so questionary owns the terminal). Presentation only; a
failure here never aborts a run (the event bus guards it).

Off a styled terminal (CI, piped) spinners and live progress rows are skipped, but the run keeps its
structure: ``output`` lines, phase headers, one line per finished agent or synth step, and the phase
summaries all still print, as plain text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.display.console import emit, fail, get_console, is_rich, ok, say, section, warn
from agent_hardener.display.context import DisplayContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

# Events that still render off a styled terminal (as plain lines); everything else is spinner/live-row only.
_PLAIN_EVENTS = frozenset(
    {
        "phase_started",
        "phase_completed",
        "agent_completed",
        "agent_failed",
        "attacker_failure_tolerated",
        "attack_summary",
        "defender_summary",
        "validator_summary",
        "synth_phase",  # one line per finished synth step
    }
)

# Longest failure reason shown on an agent's line; the full error is in the log file.
_REASON_MAX_CHARS = 160

# phase key -> (banner label, emoji, accent color). Only the war-game phases get a banner; the
# internal "victim" replay phase is intentionally omitted to keep the timeline readable.
_PHASES: dict[str, tuple[str, str, str]] = {
    "attackers": ("ATTACK", "⚔", "red"),
    "defenders": ("DEFENSE", "🛡", "cyan"),
    "validators": ("VALIDATION", "🔬", "magenta"),
}
# Phases closed by a tally of their agent runs (headline label, noun). ATTACK has its own hits summary.
_TALLIED_PHASES: dict[str, tuple[str, str]] = {
    "defenders": ("DEFENSE", "defender run"),
    "validators": ("VALIDATION", "validator"),
}


class _Spinner:
    """A one-line spinner with an elapsed clock, so a long blocking step visibly keeps moving.

    Same ``start``/``stop``/``update`` surface as ``Console.status``, which has no clock. The text is cut to the
    terminal's width: rich drops the spinner and clock columns, not the text, when a line does not fit.
    """

    _RESERVED = 12  # spinner + "0:00:00" + the gaps between columns

    def __init__(self, text: str) -> None:
        from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn  # noqa: PLC0415

        self._console = get_console()
        self._progress = Progress(
            SpinnerColumn(),
            TimeElapsedColumn(),
            TextColumn("{task.description}", markup=False),
            console=self._console,
            transient=True,
        )
        self._task = self._progress.add_task(self._fit(text), total=None)

    def start(self) -> None:
        self._progress.start()

    def stop(self) -> None:
        self._progress.stop()

    def update(self, text: str) -> None:
        self._progress.update(self._task, description=self._fit(text))

    def _fit(self, text: str) -> str:
        from rich.text import Text  # noqa: PLC0415

        fitted = Text(text)
        fitted.truncate(max(10, self._console.width - self._RESERVED), overflow="ellipsis")
        return fitted.plain


class ConsoleRenderer:
    """Stateful event subscriber that renders the run's live console narrative.

    ``output`` events and the events in ``_PLAIN_EVENTS`` print on any stream; spinners and live
    per-agent rows only on a styled terminal.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        self._ctx = DisplayContext(verbose=verbose)
        self._rich = is_rich()
        self._progress: Progress | None = None
        self._task_ids: dict[str, TaskID] = {}  # agent_id → rich task id
        self._status: Any = None  # active spinner (``_Spinner``), if any
        self._status_label = ""
        # This phase's agent results: (name, failure reason or "") -> count. Drives the phase headline and
        # keeps a failure that repeats per attack to one line off a terminal (the log has every one).
        self._results: dict[tuple[str, str], int] = {}

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "output":  # console lines always print, TTY or not
            line = str(payload.get("line", ""))
            level = payload.get("level")
            elapsed = payload.get("elapsed")
            if level:  # ``line`` is pre-formatted; style the bare ``message`` at its level instead
                message = str(payload.get("message", line))
                say(str(level), message, elapsed=float(elapsed) if isinstance(elapsed, (int, float)) else None)
            else:
                emit(line)
            return
        if not self._rich and event not in _PLAIN_EVENTS:  # spinners/live rows are styled-terminal only
            return
        handler: Callable[[dict[str, Any]], None] | None = getattr(self, f"_on_{event}", None)
        if handler is not None:
            handler(payload)

    # --- spinner for blocking ops / benign recon (was status_cm / synth_spinner) ----------

    def _on_status_started(self, payload: dict[str, Any]) -> None:
        self._stop_progress()
        self._status_label = str(payload.get("label") or "")
        detail = payload.get("detail")  # what the step is doing first, e.g. the synth's first node
        self._status = _Spinner(f"{self._status_label} — {detail}…" if detail else f"{self._status_label}…")
        self._status.start()

    def _on_status_completed(self, _payload: dict[str, Any]) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None

    def _on_synth_phase(self, payload: dict[str, Any]) -> None:
        """Print what the finished step produced and point the spinner at the step running now."""
        line = str(payload.get("label") or payload.get("phase") or "")
        if payload.get("detail"):
            line += f" — {payload['detail']}"
        seconds = payload.get("seconds")
        ok(line, elapsed=float(seconds) if isinstance(seconds, (int, float)) else None, markup=False)
        if self._status is not None and payload.get("next"):
            self._status.update(f"{self._status_label} — {payload['next']}…")

    def _on_interview_started(self, _payload: dict[str, Any]) -> None:
        if self._status is not None:
            self._status.stop()  # release the terminal while questionary prompts

    def _on_interview_completed(self, _payload: dict[str, Any]) -> None:
        if self._status is not None:
            self._status.start()

    # --- phase boundaries -----------------------------------------------------------------

    def _on_phase_started(self, payload: dict[str, Any]) -> None:
        banner = _PHASES.get(str(payload.get("phase")))
        if banner is None:
            return
        from rich.rule import Rule  # noqa: PLC0415

        self._results = {}
        label, emoji, color = banner
        count = payload.get("count")
        suffix = f"  ({count} {'agent' if count == 1 else 'agents'})" if isinstance(count, int) else ""
        if not self._rich:
            section(f"{label} {suffix.strip()}".rstrip(), markup=False)
            return
        get_console().print(Rule(f"{emoji}  {label}{suffix}", style=color))
        self._start_progress()

    def _on_phase_completed(self, payload: dict[str, Any]) -> None:
        """Close DEFENSE / VALIDATION with how many agent runs succeeded (ATTACK has its own summary)."""
        self._stop_progress()
        phase = str(payload.get("phase"))
        if phase not in _TALLIED_PHASES:
            return
        label, noun = _TALLIED_PHASES[phase]
        total = sum(self._results.values())
        if not total:  # no per-agent events seen: the phase's own count and verdict are all there is
            count = payload.get("count") or 0
            if not count:
                warn(f"{label} complete — no {noun} ran", markup=False)
                return
            (ok if payload.get("ok", True) else fail)(f"{label} complete — {count} {noun}(s)", markup=False)
            return
        succeeded = sum(n for (_name, reason), n in self._results.items() if not reason)
        repeated = total - len(self._results)
        line = f"{label} complete — {succeeded} / {total} {noun}(s) succeeded"
        if repeated and not self._rich:
            line += f"; {repeated} repeated result(s) not shown again (all are in the log)"
        (ok if succeeded == total else fail)(line, markup=False)

    # --- per-agent progress ---------------------------------------------------------------

    def _on_agent_started(self, payload: dict[str, Any]) -> None:
        if self._progress is None:
            return
        agent_id = str(payload.get("agent_id") or "")
        agent_name = str(payload.get("agent_name") or agent_id or "<agent>")
        task_id = self._progress.add_task(agent_name, total=None)
        self._task_ids[agent_id] = task_id

    def _on_agent_progress(self, payload: dict[str, Any]) -> None:
        if self._progress is None:
            return
        agent_id = str(payload.get("agent_id") or "")
        task_id = self._task_ids.get(agent_id)
        if task_id is None:
            return
        agent_name = str(payload.get("agent_name") or agent_id)
        message = str(payload.get("message") or "")
        current = payload.get("current")
        total = payload.get("total")
        description = f"{agent_name}: {message}" if message else agent_name
        self._progress.update(
            task_id,
            description=description,
            completed=current if isinstance(current, int) else None,
            total=total if isinstance(total, int) else None,
        )

    def _on_agent_completed(self, payload: dict[str, Any]) -> None:
        self._finish_agent_task(payload, succeeded=bool(payload.get("ok", True)))

    def _on_agent_failed(self, payload: dict[str, Any]) -> None:
        self._finish_agent_task(payload, succeeded=False)

    def _on_attacker_failure_tolerated(self, payload: dict[str, Any]) -> None:
        self._stop_progress()
        name = payload.get("agent_name") or "<attacker>"
        reason = _failure_reason(payload)
        warn(f"optional attacker {name} did not complete; the round goes on without it: {reason}", markup=False)

    def _finish_agent_task(self, payload: dict[str, Any], *, succeeded: bool) -> None:
        """Close an agent's row: ``✓ name`` or ``✗ name: reason`` (a plain line off a styled terminal)."""
        agent_id = str(payload.get("agent_id") or "")
        agent_name = str(payload.get("agent_name") or agent_id or "<agent>")
        reason = _failure_reason(payload) if not succeeded else ""
        note = reason or (_one_line(payload.get("summary")) if payload.get("phase") == "validators" else "")
        label = f"{agent_name}: {note}" if note else agent_name
        key = (agent_name, reason)
        seen = key in self._results
        self._results[key] = self._results.get(key, 0) + 1
        if not self._rich:
            if seen and not succeeded:
                return  # the same failure again: counted in the phase headline, and every one is in the log
            duration = payload.get("duration_seconds")
            elapsed = float(duration) if isinstance(duration, (int, float)) else None
            ok(label, elapsed=elapsed, markup=False) if succeeded else fail(label, markup=False)
            return
        task_id = self._task_ids.get(agent_id)
        if self._progress is None or task_id is None:
            return
        from rich.markup import escape  # noqa: PLC0415

        mark = "✓" if succeeded else "✗"
        style = "green" if succeeded else "red"
        self._progress.update(task_id, description=f"[{style}]{mark} {escape(label)}[/{style}]", total=1, completed=1)

    # --- headline highlights --------------------------------------------------------------

    def _print_summary(self, payload: dict[str, Any], renderer: Callable[..., tuple[Any, list[Any]]]) -> None:
        """Stop the spinner, then print a summary renderer's headline(s) followed by its detail parts."""
        self._stop_progress()
        head, parts = renderer(payload, self._ctx)
        console = get_console()
        for line in head if isinstance(head, list) else [head]:
            console.print(line)
        for part in parts:
            console.print(part)

    def _on_attack_summary(self, payload: dict[str, Any]) -> None:
        from agent_hardener.display import render_attack_summary  # noqa: PLC0415

        self._print_summary(payload, render_attack_summary)

    def _on_defender_summary(self, payload: dict[str, Any]) -> None:
        from agent_hardener.display import render_defender_summary  # noqa: PLC0415

        self._print_summary(payload, render_defender_summary)

    def _on_validator_summary(self, payload: dict[str, Any]) -> None:
        from agent_hardener.display import render_validator_summary  # noqa: PLC0415

        self._print_summary(payload, render_validator_summary)

    # --- rich progress lifecycle ----------------------------------------------------------

    def _start_progress(self) -> None:
        self._stop_progress()
        from rich.progress import Progress, SpinnerColumn, TaskID, TextColumn, TimeElapsedColumn  # noqa: PLC0415, F401

        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
            console=get_console(),
            transient=False,
        )
        self._task_ids = {}
        self._progress.start()

    def _stop_progress(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_ids = {}


def _failure_reason(payload: dict[str, Any]) -> str:
    """The one-line reason an agent failed: its ``error``, else its ``summary`` (a validator's verdict)."""
    return _one_line(payload.get("error")) or _one_line(payload.get("summary")) or "failed"


def _one_line(text: object) -> str:
    """The first line of ``text``, cut to ``_REASON_MAX_CHARS``."""
    lines = str(text or "").strip().splitlines()
    line = lines[0] if lines else ""
    return line if len(line) <= _REASON_MAX_CHARS else line[: _REASON_MAX_CHARS - 1] + "…"


def build_console_renderer(*, verbose: bool) -> ConsoleRenderer:
    """Return the console renderer used as ``run_mission``'s ``on_event`` subscriber.

    Always returns a renderer: ``output`` events print on any stream; the rich banners/spinners
    inside it no-op off a styled terminal (CI / piped), where the streamed lines and final summary
    still carry the information.
    """
    return ConsoleRenderer(verbose=verbose)
