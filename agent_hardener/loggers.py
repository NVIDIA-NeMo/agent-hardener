# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readable and structured logging for Agent Hardener sessions."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_hardener.storage import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

# A run-log subscriber receives each record both pre-formatted (for file/console sinks) and raw
# (so a UI can read ``record.levelname`` / ``record.name`` / ``record.created``).
LogSubscriber = Callable[[str, logging.LogRecord], None]

RUN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Strips ANSI escape sequences (CSI: colors, cursor moves) so styled command/console output that
# rides the bus is written as plain text to the log file. Subscribers that want color (a UI) still
# receive the original formatted line — only the file sink strips.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


class FanoutLogHandler(logging.Handler):
    """A logging handler that fans each record out to registered subscribers.

    This is the explicit, multicast log bus for an Agent Hardener run: agents and the runner are
    producers (they call ``logging.getLogger("agent_hardener...")``); the file, and later a UI, are
    subscribers registered via :meth:`subscribe`. ``logging`` serializes ``emit`` under the
    handler's lock, so the fan-out (and each subscriber's write) is atomic per record.
    """

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: list[LogSubscriber] = []

    def subscribe(self, fn: LogSubscriber) -> Callable[[], None]:
        """Register ``fn`` to receive every record; returns a no-arg unsubscribe handle."""
        self._subscribers.append(fn)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(fn)

        return _unsubscribe

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        for sub in list(self._subscribers):
            try:
                sub(msg, record)
            except Exception:  # a bad subscriber must never break the run
                # Use sys.stderr-free path: route through the session logger's own debug channel.
                logging.getLogger("agent_hardener.session").debug("log subscriber failed", exc_info=True)


@contextlib.contextmanager
def capture_run_log(path: Path) -> Iterator[FanoutLogHandler]:
    """Attach a run-scoped :class:`FanoutLogHandler` to the ``agent_hardener`` logger.

    Routes every ``agent_hardener.*`` record (agents + the runner's ``agent_hardener.run`` logger) to the
    bus for the duration of the run. A built-in subscriber writes each formatted record to ``path``
    (truncated on entry, fresh per run) — it is the only writer to that file. Yields the handler so
    callers can register additional subscribers (e.g. a UI). On exit the handler is detached and the
    file closed, leaving no global logging state behind.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = FanoutLogHandler()
    handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT))
    handler.setLevel(logging.INFO)

    # The ``agent_hardener`` ancestor logger is the bus's anchor. We make it terminal for the run
    # (propagate=False) so records stop here after the fan-out instead of continuing to the Python
    # root, where ``logging.lastResort`` would echo WARNING+ to stderr and double-print warnings.
    root = logging.getLogger("agent_hardener")
    previous_level, previous_propagate = root.level, root.propagate
    if not root.isEnabledFor(logging.INFO):
        root.setLevel(logging.INFO)
    root.propagate = False

    with path.open("w", encoding="utf-8") as file_handle:

        def _write_to_file(line: str, _record: logging.LogRecord) -> None:
            clean = _ANSI_RE.sub("", line)
            file_handle.write(clean if clean.endswith("\n") else clean + "\n")
            file_handle.flush()

        handler.subscribe(_write_to_file)
        root.addHandler(handler)
        try:
            yield handler
        finally:
            root.removeHandler(handler)
            root.setLevel(previous_level)
            root.propagate = previous_propagate
            handler.close()


def build_round_logger(round_id: str, round_dir: Path, *, quiet: bool = False) -> logging.Logger:
    """Create a console and file logger scoped to one round.

    ``quiet`` raises the console handler to WARNING (the file always keeps full INFO), so the CLI can
    suppress per-agent INFO chatter — e.g. the guardrails defender logging attack prompts — unless the
    user passes ``--verbose``. Real problems (WARNING/ERROR) still reach the console.

    ``propagate`` is left on so records also bubble to the ``agent_hardener`` run-log bus (see
    :func:`capture_run_log`); no console handler exists at the ancestor level, so this cannot
    double-print to the terminal.
    """
    logger = logging.getLogger("agent_hardener.session")
    logger.setLevel(logging.INFO)
    logger.propagate = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.WARNING if quiet else logging.INFO)
    file_handler = logging.FileHandler(round_dir / "round.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


class JsonlEventLogger:
    """Append-only structured event log with thread-safe writes."""

    def __init__(self, path: Path, observer: Callable[[str, dict[str, Any]], None] | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        # Optional live subscriber (e.g. the CLI phase reporter). Never lets a display error break a run.
        self._observer = observer

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **payload,
        }
        line = json.dumps(to_jsonable(record), sort_keys=True) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()
        if self._observer is not None:
            try:
                self._observer(event, payload)
            except Exception:
                logging.getLogger("agent_hardener.session").debug("event observer failed for %s", event, exc_info=True)

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        with self._lock:
            self._file.flush()
            self._file.close()


# An event subscriber receives ``(event_name, payload)`` for each emitted structured event.
EventSubscriber = Callable[[str, "dict[str, Any]"], None]

# Ambient run-scoped event sink. The runner sets it (via :func:`using_event_sink`) to the run's
# EventBus.emit for the whole run, so producers — the orchestrator's per-session event loggers and
# the pre-flight synth alike — emit through one channel without threading a callback. Mirrors how the
# global ``agent_hardener`` logger reaches the run-scoped log bus.
_event_sink: contextvars.ContextVar[EventSubscriber | None] = contextvars.ContextVar(
    "agent_hardener_event_sink", default=None
)


@contextlib.contextmanager
def using_event_sink(emit: EventSubscriber) -> Iterator[None]:
    """Bind the ambient event sink to ``emit`` for the duration of the block."""
    token = _event_sink.set(emit)
    try:
        yield
    finally:
        _event_sink.reset(token)


def emit_event(event: str, payload: dict[str, Any]) -> None:
    """Emit a structured event to the ambient sink; a no-op when no sink is active."""
    sink = _event_sink.get()
    if sink is not None:
        sink(event, payload)


# The agent currently executing (agent_id/name/role/validator_kind), set by the agent stages around each
# invocation so the LLM telemetry callback can attribute every model call to the right agent.
_current_agent: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "agent_hardener_current_agent", default=None
)


@contextlib.contextmanager
def using_current_agent(agent: dict[str, Any]) -> Iterator[None]:
    """Bind the ambient current-agent identity for the duration of the block."""
    token = _current_agent.set(agent)
    try:
        yield
    finally:
        _current_agent.reset(token)


def current_agent() -> dict[str, Any] | None:
    """The agent identity currently executing, or None outside an agent invocation."""
    return _current_agent.get()


class EventBus:
    """Run-scoped, multicast structured-event bus.

    Fans each event out to subscribers — the CLI console renderer, the events→log bridge, and a future
    UI. :meth:`emit` matches the ``(event, payload)`` observer signature, so it is
    what the ambient sink and the orchestrator's per-session loggers forward into. Persistence is the
    per-round ``events.jsonl`` (see :func:`round_logging`); the bus itself is in-memory fan-out only.
    """

    def __init__(self) -> None:
        self._subscribers: list[EventSubscriber] = []

    def subscribe(self, fn: EventSubscriber) -> Callable[[], None]:
        """Register ``fn`` to receive every event; returns a no-arg unsubscribe handle."""
        self._subscribers.append(fn)

        def _unsubscribe() -> None:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(fn)

        return _unsubscribe

    def emit(self, event: str, payload: dict[str, Any]) -> None:
        """Fan the event out to subscribers (persistence lives in the per-round events.jsonl)."""
        for fn in list(self._subscribers):
            try:
                fn(event, payload)
            except Exception:  # a bad subscriber must never break the run
                logging.getLogger("agent_hardener.session").debug(
                    "event subscriber failed for %s", event, exc_info=True
                )


@contextlib.contextmanager
def round_logging(round_id: str, round_dir: Path) -> Iterator[tuple[logging.Logger, JsonlEventLogger]]:
    """Per-round logger + structured-event logger, torn down on exit.

    Console stays quiet (WARNING+) so parallel agents' live INFO never interleaves; the CLI's phase
    reporter prints the ordered conversation at each phase end, while round.log keeps full INFO. The
    per-round events.jsonl forwards every event to the ambient run-scoped sink (the run's EventBus) so
    phase/attack/defender events reach live subscribers and the run-level events.jsonl; ``emit_event``
    is a no-op outside a run (e.g. isolated tests), so events still persist.
    """
    logger = build_round_logger(round_id, round_dir, quiet=True)
    events = JsonlEventLogger(round_dir / "events.jsonl", observer=emit_event)
    try:
        yield logger, events
    finally:
        events.close()
        for handler in logger.handlers[:]:
            handler.flush()
            logger.removeHandler(handler)
            handler.close()
