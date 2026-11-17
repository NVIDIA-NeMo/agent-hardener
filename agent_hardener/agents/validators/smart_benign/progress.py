# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live progress for the synth DAG, so a minutes-long ``synth-benign`` run never looks stuck.

Each finished node prints a ``✓`` line with what it produced and how long it took; on a styled terminal a
spinner below them names the node running now and counts the elapsed time. The spinner is paused while the
interview owns the terminal. Off a TTY (CI, piped) only the ``✓`` lines print.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from agent_hardener.display.console import get_console, is_rich, step

from .graph import _ingestion_sources, _route_after_gap_detector
from .state import SynthState

if TYPE_CHECKING:
    from types import TracebackType

    from rich.progress import Progress, TaskID

    from .driver import AnswerProvider

#: What each synth-DAG node did, keyed by the node names registered in ``graph.py``.
PHASE_LABELS: dict[str, str] = {
    "nl_parser": "parsed capabilities from the description",
    "github_analyzer": "analyzed the GitHub repository",
    "api_prober": "probed the victim endpoint for tools",
    "profile_synthesizer": "merged the victim capability profile",
    "gap_detector": "checked the profile for gaps",
    "interview_compose": "composed the interview questions",
    "interview_ask": "recorded the interview answers",
    "request_generator": "generated the benign request suite",
    "critic": "filtered and deduplicated requests",
    "profile_writer": "wrote profile.json + requests.csv",
}

# What a node is doing while it runs: the spinner shows this for the node that runs next.
_RUNNING: dict[str, str] = {
    "nl_parser": "parsing the description",
    "github_analyzer": "analyzing the GitHub repository",
    "api_prober": "probing the victim endpoint",
    "profile_synthesizer": "merging the victim capability profile",
    "gap_detector": "checking the profile for gaps",
    "interview_compose": "composing interview questions",
    "interview_ask": "waiting for interview answers",
    "request_generator": "generating the benign request suite",
    "critic": "filtering and deduplicating requests",
    "profile_writer": "writing profile.json + requests.csv",
}

# Ingestion node → its key in ``SynthState.partials``.
_PARTIAL_KEYS = {"nl_parser": "nl", "github_analyzer": "github", "api_prober": "api_probe"}

# Short names for the ingestion sources, which run in parallel and so share one spinner line.
_SOURCES = {"nl_parser": "the description", "github_analyzer": "GitHub repo", "api_prober": "victim endpoint"}

_TITLE = "Synthesizing benign suite"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _duration(seconds: float) -> str:
    whole = round(seconds)
    return f"{whole}s" if whole < 60 else f"{whole // 60}m {whole % 60:02d}s"


def ok(message: str) -> None:
    """Print a finished-step line; never wrapped, so a piped log (80 columns by default) keeps one line per step."""
    get_console().print(f"✓ {message}", style="green" if is_rich() else None, highlight=False, soft_wrap=True)


def _merge(state: SynthState, update: dict[str, Any]) -> SynthState:
    """Apply one node's update the way the graph does: reducer fields accumulate, the rest overwrite."""
    merged: dict[str, Any] = {}
    for key, value in update.items():
        field = SynthState.model_fields[key]  # the graph rejects updates outside its state schema
        reducer = next((m for m in field.metadata if callable(m)), None)
        merged[key] = reducer(getattr(state, key), value) if reducer else value
    return state.model_copy(update=merged)


class SynthProgress:
    """Context manager that renders synth progress from ``drive_synth``'s ``on_node`` callback.

    Tracks a mirror of the graph state so each ``✓`` line can say what the node produced (tools, gaps,
    requests kept by the critic) and the spinner can name the node the graph routes to next.
    """

    def __init__(self, state: SynthState) -> None:
        self._state = state
        self._ingesting = {s for s in _ingestion_sources(state) if s in _PARTIAL_KEYS}
        self._started = self._last = 0.0
        self._progress: Progress | None = None
        self._task: TaskID | None = None

    def __enter__(self) -> SynthProgress:
        """Start the clock and the spinner (or print a plain start line off a styled terminal)."""
        self._started = self._last = time.monotonic()
        running = self._pending_ingestion() or _RUNNING["profile_synthesizer"]
        if is_rich():
            from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn  # noqa: PLC0415

            self._progress = Progress(
                SpinnerColumn(),
                TimeElapsedColumn(),  # before the text, so a narrow terminal truncates the text, never the clock
                TextColumn("{task.description}"),
                console=get_console(),
                transient=True,
            )
            self._task = self._progress.add_task(f"{_TITLE} — {running}…", total=None)
            self._progress.start()
        else:
            step(f"{_TITLE}…")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, _exc: BaseException | None, _tb: TracebackType | None
    ) -> None:
        """Stop the spinner; print the total time only when synthesis finished without raising."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
        if exc_type is None:
            ok(f"Benign suite synthesized in {_duration(time.monotonic() - self._started)}")

    def on_node(self, name: str, update: dict[str, Any]) -> None:
        """Print the finished node's line and point the spinner at what runs next."""
        if name not in PHASE_LABELS:
            return
        now = time.monotonic()
        # Ingestion nodes run in parallel from the start, so each one's time is measured from there.
        took = now - (self._started if name in self._ingesting else self._last)
        self._last = now
        before = self._state
        self._state = _merge(before, update)
        self._ingesting.discard(name)

        counts = self._counts(name, update, before)
        timing = "" if name == "interview_ask" else f" ({_duration(took)})"  # mostly the operator's think time
        ok(f"{PHASE_LABELS[name]}{f' — {counts}' if counts else ''}{timing}")
        self._set_running(self._next(name))

    def pausing(self, provider: AnswerProvider) -> AnswerProvider:
        """Wrap an answer provider so the spinner releases the terminal while it runs."""

        async def paused(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
            if self._progress is not None:
                self._progress.stop()
            try:
                return await provider(questions)
            finally:
                if self._progress is not None:
                    self._progress.start()

        return paused

    def _counts(self, name: str, update: dict[str, Any], before: SynthState) -> str:
        parts: list[str] = []
        after = self._state
        if name in _PARTIAL_KEYS:
            parts.append(_plural(len((update.get("partials") or {}).get(_PARTIAL_KEYS[name], [])), "tool"))
            if name == "api_prober":
                probe = (update.get("source_details") or {}).get("api_probe") or {}
                if probe_errors := len(probe.get("probe_errors") or []):
                    parts.append(_plural(probe_errors, "probe error"))
        elif name == "profile_synthesizer" and after.profile is not None:
            parts.append(_plural(len(after.profile.tools), "tool"))
            parts.append(_plural(len(after.profile.personas), "persona"))
            parts.append(f"{len(after.profile.out_of_scope)} out-of-scope")
        elif name == "gap_detector":
            answered = {gap for gap, _q, _a in after.interviewer_answers if gap}
            unresolved = [g for g in after.gaps if g not in answered]
            parts.append(f"{_plural(len(unresolved), 'gap')} unresolved")
        elif name == "interview_compose":
            parts.append(_plural(len(after.composed_questions), "question"))
        elif name == "interview_ask" and before.composed_questions:
            parts.append(f"{len(update.get('interviewer_answers') or [])} of {len(before.composed_questions)} answered")
        elif name == "request_generator":
            parts.append(_plural(len(after.requests), "request"))
        elif name == "critic":
            parts.append(f"{len(before.requests)} → {len(after.requests)} kept")
        if errors := len(update.get("errors") or []):
            parts.append(_plural(errors, "error"))
        return ", ".join(parts)

    def _next(self, name: str) -> str:
        if name in _PARTIAL_KEYS:
            return self._pending_ingestion() or _RUNNING["profile_synthesizer"]
        if name == "gap_detector":
            return _RUNNING[_route_after_gap_detector(self._state)]
        if name == "interview_ask":
            return "re-merging the profile with the answers"
        following = {
            "profile_synthesizer": "gap_detector",
            "interview_compose": "interview_ask",
            "request_generator": "critic",
            "critic": "profile_writer",
        }.get(name)
        return _RUNNING[following] if following else "finishing"

    def _pending_ingestion(self) -> str:
        """The ingestion nodes still running, in pipeline order (empty once they have all finished)."""
        pending = [_SOURCES[n] for n in _PARTIAL_KEYS if n in self._ingesting]
        return f"reading {', '.join(pending)}" if pending else ""

    def _set_running(self, running: str) -> None:
        if self._progress is not None and self._task is not None:
            self._progress.update(self._task, description=f"{_TITLE} — {running}…")
