# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readable and structured logging for Agent Hardener sessions."""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from agent_hardener.storage import to_jsonable

if TYPE_CHECKING:
    from collections.abc import Iterator

# A run-log subscriber receives each record both pre-formatted (for file/console sinks) and raw
# (so a UI can read ``record.levelname`` / ``record.name`` / ``record.created``).
LogSubscriber = Callable[[str, logging.LogRecord], None]

RUN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"

# Strips ANSI escape sequences (CSI: colors, cursor moves) so styled command/console output that
# rides the bus is written as plain text to the log file. Subscribers that want color (a UI) still
# receive the original formatted line — only the file sink strips.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")

# Where every CLI invocation writes its full diagnostic log: ``<cwd>/.agent-hardener/logs/<command>-<ts>.log``.
CLI_LOGS_DIR = Path(".agent-hardener") / "logs"
#: Names the CLI log file outright (instead of ``CLI_LOGS_DIR/<command>-<time>.log``).
LOG_FILE_ENV = "AGENT_HARDENER_LOG_FILE"

# ``extra=FILE_ONLY`` keeps a record out of the console even under ``--debug`` — for blobs that already have
# their own console rendering (the final report) and would otherwise print twice.
FILE_ONLY: dict[str, Any] = {"console": False}

# The events→log bridge echoes every run event (including each ``output`` line the console already printed)
# into the log; the console renders those events itself, so this logger never reaches it.
_EVENTS_LOGGER = "agent_hardener.events"

# One console line at most this long; the full message is in the log file.
_CONSOLE_MAX_CHARS = 300

# Marks the handlers ``configure_cli_logging`` installs, so a second call replaces rather than stacks them.
_CLI_HANDLER_ATTR = "_agent_hardener_cli"


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
    handler.setLevel(logging.DEBUG)

    # The ``agent_hardener`` ancestor logger is the bus's anchor. We make it terminal for the run
    # (propagate=False) so records stop here after the fan-out instead of continuing to the Python
    # root, where ``logging.lastResort`` would echo WARNING+ to stderr and double-print warnings.
    root = logging.getLogger("agent_hardener")
    previous_level, previous_propagate = root.level, root.propagate
    if not root.isEnabledFor(logging.DEBUG):
        root.setLevel(logging.DEBUG)
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


def build_round_logger(round_id: str, round_dir: Path) -> logging.Logger:
    """Create the file logger scoped to one round (``round.log``).

    It has no console handler of its own: records propagate to the ``agent_hardener`` logger, where the
    run-log bus (see :func:`capture_run_log`) writes them to ``agent-hardener.log`` and, in the CLI, the
    console handler from :func:`configure_cli_logging` prints warnings as one-line ``!``/``✗`` messages.
    A raw stream handler here used to print timestamped records and tracebacks straight over the live
    progress display.
    """
    logger = logging.getLogger("agent_hardener.session")
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(round_dir / "round.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT))
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger


class ConsoleLogHandler(logging.Handler):
    """Render log records on the CLI console in the same vocabulary as every other line.

    By default only WARNING+ reach the console, as one line: ``! message`` or ``✗ message``. The message is
    cut to its first line, tracebacks are dropped and ``(details in log)`` points at the file that has them.
    With ``--debug`` every record prints in the log format (time, level, logger, message, traceback), dimmed
    below WARNING. Records go through the shared rich console, so they print above a live spinner or
    progress display instead of tearing it.
    """

    _DEBUG_STYLES: ClassVar[dict[int, str]] = {logging.DEBUG: "dim", logging.INFO: "dim", logging.WARNING: "yellow"}

    def __init__(self, *, debug: bool) -> None:
        super().__init__(logging.DEBUG if debug else logging.WARNING)
        self._debug = debug
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))

    def filter(self, record: logging.LogRecord) -> bool:
        """Drop file-only records: the event echo and anything logged with ``extra=FILE_ONLY``."""
        if record.name == _EVENTS_LOGGER or not getattr(record, "console", True):
            return False
        return super().filter(record)

    def emit(self, record: logging.LogRecord) -> None:
        # Imported here: ``display`` pulls in models and renderers, and this module is imported by the engine.
        from agent_hardener.display.console import fail, literal, warn  # noqa: PLC0415

        try:
            if self._debug:
                short = logging.makeLogRecord({**record.__dict__, "name": record.name.removeprefix("agent_hardener.")})
                literal(self.format(short), self._DEBUG_STYLES.get(record.levelno, "red"))
                return
            message = record.getMessage().strip()
            first_line, _, rest = message.partition("\n")
            if len(first_line) > _CONSOLE_MAX_CHARS:
                first_line = first_line[: _CONSOLE_MAX_CHARS - 1] + "…"
            if record.exc_info or rest:
                first_line += " (details in log)"
            (fail if record.levelno >= logging.ERROR else warn)(first_line, markup=False)
        except Exception:
            self.handleError(record)


class _LazyFileHandler(logging.FileHandler):
    """A file handler that creates its directory and file only when the first record arrives.

    So ``agent-hardener --help`` or a command that logs nothing leaves no empty log file behind.
    """

    def __init__(self, path: Path) -> None:
        super().__init__(path, encoding="utf-8", delay=True)

    def _open(self):  # type: ignore[no-untyped-def]
        Path(self.baseFilename).parent.mkdir(parents=True, exist_ok=True)
        return super()._open()


def configure_cli_logging(command: str, *, debug: bool, logs_dir: Path | None = None) -> Path:
    """Wire logging for one CLI invocation and return the path of its diagnostic log file.

    * The log file gets every ``agent_hardener.*`` record at DEBUG and third-party records at WARNING
      (INFO with ``--debug``), with tracebacks. It is created on the first record.
    * The console gets ``agent_hardener.*`` WARNING+ as one-line ``!``/``✗`` messages (everything, in the log
      format, with ``--debug``). Third-party libraries reach the console only with ``--debug``.

    ``AGENT_HARDENER_LOG_FILE`` names the file instead, for a caller that keeps it somewhere other than the
    working directory (Helix uploads it; only a job's stdout survives otherwise).

    The ``agent_hardener`` logger is made terminal (``propagate=False``) so its records are not seen twice by
    the root handlers. Calling it again replaces the handlers it installed.
    """
    from agent_hardener.display.console import set_debug  # noqa: PLC0415 - see ConsoleLogHandler.emit

    set_debug(debug)
    if chosen := os.environ.get(LOG_FILE_ENV):
        path = Path(chosen).expanduser()
    else:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        path = (logs_dir or Path.cwd() / CLI_LOGS_DIR) / f"{command}-{stamp}.log"

    file_handler = _LazyFileHandler(path)
    file_handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT))
    file_handler.setLevel(logging.DEBUG)
    console_handler = ConsoleLogHandler(debug=debug)
    for handler in (file_handler, console_handler):
        setattr(handler, _CLI_HANDLER_ATTR, True)

    package = logging.getLogger("agent_hardener")
    root = logging.getLogger()
    for logger in (package, root):
        for old in [h for h in logger.handlers if getattr(h, _CLI_HANDLER_ATTR, False)]:
            logger.removeHandler(old)
            old.close()

    package.setLevel(logging.DEBUG)
    package.propagate = False
    package.addHandler(file_handler)
    package.addHandler(console_handler)

    root.setLevel(logging.INFO if debug else logging.WARNING)
    root.addHandler(file_handler)
    if debug:
        root.addHandler(console_handler)
    return path


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

    round.log keeps full INFO; the console sees the round's records only through the CLI's console
    handler (see :func:`configure_cli_logging`). The per-round events.jsonl forwards every event to the ambient run-scoped sink (the run's EventBus) so
    phase/attack/defender events reach live subscribers and the run-level events.jsonl; ``emit_event``
    is a no-op outside a run (e.g. isolated tests), so events still persist.
    """
    logger = build_round_logger(round_id, round_dir)
    events = JsonlEventLogger(round_dir / "events.jsonl", observer=emit_event)
    try:
        yield logger, events
    finally:
        events.close()
        for handler in logger.handlers[:]:
            handler.flush()
            logger.removeHandler(handler)
            handler.close()
