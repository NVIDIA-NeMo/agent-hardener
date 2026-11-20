# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live console renderer: turns run events into colored, phase-structured console output.

Wired by the CLI as the sole ``on_event`` subscriber. All one-way presentation flows through here as
events: ``output`` highlight lines, ``status_started``/``status_completed`` spinners with an elapsed clock
(sandbox bring-up, health, benign synthesis), ATTACK / DEFENSE / VALIDATION banners, live per-agent progress,
and benign synthesis steps (each ``synth_phase`` prints a line with what the step produced and moves the
spinner to the step running now; the spinner is paused around the interview via
``interview_started``/``interview_completed`` so questionary owns the terminal). Presentation only; a
failure here never aborts a run (the event bus guards it). ``output`` and ``synth_phase`` always print (even
off a TTY); the rich banners/spinners no-op when not on a styled terminal.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from agent_hardener.display.console import emit, get_console, is_rich
from agent_hardener.display.context import DisplayContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.progress import Progress, TaskID

# phase key -> (banner label, emoji, accent color). Only the war-game phases get a banner; the
# internal "victim" replay phase is intentionally omitted to keep the timeline readable.
_PHASES: dict[str, tuple[str, str, str]] = {
    "attackers": ("ATTACK", "⚔", "red"),
    "defenders": ("DEFENSE", "🛡", "cyan"),
    "validators": ("VALIDATION", "🔬", "magenta"),
}


def format_duration(seconds: float) -> str:
    """``4s`` under a minute, ``2m 05s`` past one."""
    whole = math.floor(seconds + 0.5)
    return f"{whole}s" if whole < 60 else f"{whole // 60}m {whole % 60:02d}s"


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

    ``output`` events always print (so CI/piped runs still get highlight lines); every other (rich)
    event is rendered only on a styled terminal.
    """

    def __init__(self, *, verbose: bool = False) -> None:
        self._ctx = DisplayContext(verbose=verbose)
        self._rich = is_rich()
        self._progress: Progress | None = None
        self._task_ids: dict[str, TaskID] = {}  # agent_id → rich task id
        self._status: Any = None  # active spinner (``_Spinner``), if any
        self._status_label = ""

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        if event == "output":  # highlight lines always print, TTY or not
            emit(str(payload.get("line", "")))
            return
        if event == "synth_phase":  # one line per finished synth step, TTY or not
            self._on_synth_phase(payload)
            return
        if not self._rich:  # banners/spinners/summaries are styled-terminal only
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
        if isinstance(seconds, (int, float)):
            line += f" ({format_duration(seconds)})"
        # Never wrapped: a piped log (80 columns by default) keeps one line per step.
        get_console().print(
            f"✓ {line}", style="green" if self._rich else None, highlight=False, markup=False, soft_wrap=True
        )
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

        label, emoji, color = banner
        count = payload.get("count")
        suffix = f"  ({count} {'agent' if count == 1 else 'agents'})" if isinstance(count, int) else ""
        get_console().print(Rule(f"{emoji}  {label}{suffix}", style=color))
        self._start_progress()

    def _on_phase_completed(self, payload: dict[str, Any]) -> None:
        self._stop_progress()
        if payload.get("phase") != "validators":
            return
        from rich.text import Text  # noqa: PLC0415

        count = payload.get("count", 0)
        ok = payload.get("ok", True)
        line = Text("✓ VALIDATION complete — ")
        line.append(f"{count} validator{'' if count == 1 else 's'}", style="green" if ok else "red")
        get_console().print(line)

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
        self._finish_agent_task(payload, ok=True)

    def _on_agent_failed(self, payload: dict[str, Any]) -> None:
        self._finish_agent_task(payload, ok=False)

    def _finish_agent_task(self, payload: dict[str, Any], *, ok: bool) -> None:
        if self._progress is None:
            return
        agent_id = str(payload.get("agent_id") or "")
        task_id = self._task_ids.get(agent_id)
        if task_id is None:
            return
        agent_name = str(payload.get("agent_name") or agent_id)
        mark = "✓" if ok else "✗"
        style = "green" if ok else "red"
        self._progress.update(task_id, description=f"[{style}]{mark} {agent_name}[/{style}]", total=1, completed=1)

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


def build_console_renderer(*, verbose: bool) -> ConsoleRenderer:
    """Return the console renderer used as ``run_mission``'s ``on_event`` subscriber.

    Always returns a renderer: ``output`` events print on any stream; the rich banners/spinners
    inside it no-op off a styled terminal (CI / piped), where the streamed lines and final summary
    still carry the information.
    """
    return ConsoleRenderer(verbose=verbose)
