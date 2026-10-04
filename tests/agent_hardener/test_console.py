# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the shared terminal presentation primitives."""

from __future__ import annotations

import contextlib
import io

import pytest
from rich.console import Console

from agent_hardener.display import console

pytestmark = pytest.mark.unit


def _terminal_console() -> Console:
    return Console(file=io.StringIO(), force_terminal=True, width=80)


def _use_console(monkeypatch: pytest.MonkeyPatch, con: Console) -> None:
    """Point the module at ``con`` and clear the plain-output env gates."""
    monkeypatch.setattr(console, "get_console", lambda: con)
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_PLAIN", raising=False)


def test_get_console_returns_cached_console() -> None:
    assert isinstance(console.get_console(), Console)
    assert console.get_console() is console.get_console()  # cached singleton


# --- is_rich gating -----------------------------------------------------------------------


def test_is_rich_true_on_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_console(monkeypatch, _terminal_console())
    assert console.is_rich() is True


def test_is_rich_false_without_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_console(monkeypatch, Console(file=io.StringIO(), force_terminal=False))
    assert console.is_rich() is False


def test_is_rich_respects_no_color(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_console(monkeypatch, _terminal_console())
    monkeypatch.setenv("NO_COLOR", "1")
    assert console.is_rich() is False


def test_is_rich_respects_agent_hardener_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_console(monkeypatch, _terminal_console())
    monkeypatch.setenv("AGENT_HARDENER_PLAIN", "1")
    assert console.is_rich() is False


# --- glyph helpers ------------------------------------------------------------------------


def test_glyph_helpers_render_styled(monkeypatch: pytest.MonkeyPatch) -> None:
    con = _terminal_console()
    _use_console(monkeypatch, con)
    console.step("building")
    console.ok("done")
    console.fail("nope")
    console.info("plain")
    out = con.file.getvalue()
    assert "▸ building" in out
    assert "✓ done" in out
    assert "✗ nope" in out
    assert "plain" in out
    assert "\x1b[" in out  # ANSI styling emitted on a terminal


def test_helpers_plain_when_not_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    con = Console(file=io.StringIO(), force_terminal=False, width=80)
    _use_console(monkeypatch, con)
    console.ok("done")
    out = con.file.getvalue()
    assert "✓ done" in out
    assert "\x1b[" not in out  # no color codes off a terminal


# --- emit sink ----------------------------------------------------------------------------


def test_emit_styles_by_cue(monkeypatch: pytest.MonkeyPatch) -> None:
    con = _terminal_console()
    _use_console(monkeypatch, con)
    console.emit("sandbox failed to start")  # red branch
    console.emit("health check ready")  # green branch
    console.emit("just some text")  # plain branch
    out = con.file.getvalue()
    assert "sandbox failed to start" in out
    assert "health check ready" in out
    assert "just some text" in out


def test_emit_styles_each_line_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    con = _terminal_console()
    _use_console(monkeypatch, con)
    # A benign teardown line lives in the same blob as plain status lines.
    console.emit("sandbox ready: demo\nfailed to stop managed forward: Operation not permitted\nbenign tail")
    lines = con.file.getvalue().splitlines()
    by_text = {"ready": "", "failed": "", "tail": ""}
    for rendered in lines:
        if "sandbox ready" in rendered:
            by_text["ready"] = rendered
        elif "failed to stop" in rendered:
            by_text["failed"] = rendered
        elif "benign tail" in rendered:
            by_text["tail"] = rendered
    assert "\x1b[31m" in by_text["failed"]  # only the failure line is red
    assert "\x1b[31m" not in by_text["tail"]  # the rest stays plain
    assert "\x1b[31m" not in by_text["ready"]


def test_emit_does_not_redden_benign_zero_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    con = _terminal_console()
    _use_console(monkeypatch, con)
    console.emit("benign run: 0 errors, no errors found")
    assert "\x1b[31m" not in con.file.getvalue()


# --- status factory -----------------------------------------------------------------------


def test_status_factory_nullcontext_when_plain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_HARDENER_PLAIN", "1")
    cm = console.status_factory("Working")
    assert isinstance(cm, contextlib.nullcontext)


def test_status_factory_status_when_rich(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_console(monkeypatch, _terminal_console())
    cm = console.status_factory("Working")
    with cm:  # rich Status is a context manager
        pass
    assert cm.__class__.__name__ == "Status"
