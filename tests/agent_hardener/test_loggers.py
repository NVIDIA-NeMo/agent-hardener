# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the run-log fan-out bus (FanoutLogHandler + capture_run_log)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from agent_hardener.loggers import EventBus, FanoutLogHandler, capture_run_log, emit_event, using_event_sink

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _record(msg: str, *, name: str = "agent_hardener.x", level: int = logging.INFO) -> logging.LogRecord:
    return logging.makeLogRecord({"msg": msg, "name": name, "levelno": level, "levelname": logging.getLevelName(level)})


def test_fanout_delivers_to_all_subscribers_and_isolates_failures() -> None:
    handler = FanoutLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    a: list[str] = []
    b: list[str] = []

    def boom(_line: str, _rec: logging.LogRecord) -> None:
        raise RuntimeError("subscriber blew up")

    handler.subscribe(lambda line, _rec: a.append(line))
    handler.subscribe(boom)  # must not stop the others or propagate
    handler.subscribe(lambda _line, rec: b.append(rec.getMessage()))

    handler.emit(_record("hello"))

    assert a == ["hello"]
    assert b == ["hello"]


def test_subscribe_returns_working_unsubscribe() -> None:
    handler = FanoutLogHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    seen: list[str] = []
    unsubscribe = handler.subscribe(lambda line, _rec: seen.append(line))

    handler.emit(_record("one"))
    unsubscribe()
    handler.emit(_record("two"))

    assert seen == ["one"]


def test_capture_run_log_attaches_writes_records_and_detaches(tmp_path: Path) -> None:
    log_path = tmp_path / "agent-hardener.log"
    iron = logging.getLogger("agent_hardener")
    handlers_before = list(iron.handlers)
    extra: list[str] = []

    with capture_run_log(log_path) as bus:
        bus.subscribe(lambda line, _rec: extra.append(line))  # a second subscriber proves multicast
        assert any(isinstance(h, FanoutLogHandler) for h in iron.handlers)
        logging.getLogger("agent_hardener.agents.test").info("agent says hi")

    # Handler fully detached on exit — no global logging state leaks across runs.
    assert list(iron.handlers) == handlers_before

    text = log_path.read_text(encoding="utf-8")
    assert "INFO" in text
    assert "agent_hardener.agents.test" in text
    assert "agent says hi" in text
    # The file subscriber and the test's extra subscriber both received the record.
    assert extra
    assert "agent says hi" in extra[0]


def test_file_subscriber_strips_ansi_but_subscribers_keep_it(tmp_path: Path) -> None:
    log_path = tmp_path / "agent-hardener.log"
    styled = "\x1b[1m\x1b[32m✓\x1b[39m\x1b[0m Deleted sandbox"
    received: list[str] = []

    with capture_run_log(log_path) as bus:
        bus.subscribe(lambda line, _rec: received.append(line))
        logging.getLogger("agent_hardener.run").info(styled)

    text = log_path.read_text(encoding="utf-8")
    assert "\x1b[" not in text  # file is clean plain text
    assert "✓ Deleted sandbox" in text
    # Non-file subscribers (e.g. a UI) still get the original styled line.
    assert any("\x1b[" in line for line in received)


def test_event_bus_multicasts_and_isolates_failures() -> None:
    bus = EventBus()
    a: list[tuple[str, dict]] = []
    b: list[tuple[str, dict]] = []

    def boom(_event: str, _payload: dict) -> None:
        raise RuntimeError("subscriber blew up")

    bus.subscribe(lambda e, p: a.append((e, p)))
    bus.subscribe(boom)  # must not stop the others
    bus.subscribe(lambda e, p: b.append((e, p)))

    bus.emit("phase_started", {"phase": "attackers", "count": 1})

    # In-memory fan-out only: every subscriber receives the event; persistence is the per-round
    # events.jsonl, not the bus.
    assert a == [("phase_started", {"phase": "attackers", "count": 1})]
    assert b == [("phase_started", {"phase": "attackers", "count": 1})]


def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    seen: list[str] = []
    unsubscribe = bus.subscribe(lambda e, _p: seen.append(e))

    bus.emit("one", {})
    unsubscribe()
    bus.emit("two", {})

    assert seen == ["one"]


def test_emit_event_forwards_to_active_sink_and_noops_when_unset() -> None:
    seen: list[tuple[str, dict]] = []
    emit_event("ignored", {"x": 1})  # no sink set → no-op (must not raise)
    with using_event_sink(lambda e, p: seen.append((e, p))):
        emit_event("synth_phase", {"phase": "gap_detector", "label": "checking"})
    emit_event("after", {})  # sink reset on exit → no-op
    assert seen == [("synth_phase", {"phase": "gap_detector", "label": "checking"})]


def test_capture_run_log_is_readable_and_grows_while_open(tmp_path: Path) -> None:
    # A console viewer / tail -f reading the log while the run writes must see flushed rows: the
    # writer holds no exclusive lock and flushes per line.
    log_path = tmp_path / "agent-hardener.log"
    with capture_run_log(log_path):
        logging.getLogger("agent_hardener.run").info("first line")
        with log_path.open(encoding="utf-8") as reader:  # separate handle, writer still open
            assert "first line" in reader.read()
        size1 = log_path.stat().st_size
        logging.getLogger("agent_hardener.run").info("second line")
        assert log_path.stat().st_size > size1  # appended live, not buffered until close
        with log_path.open(encoding="utf-8") as reader:
            assert "second line" in reader.read()
