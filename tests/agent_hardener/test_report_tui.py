# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the tabbed verbose report: section builders, the Textual TUI, and the plain fallback."""

from __future__ import annotations

import asyncio
import io

import pytest
from rich.console import Console
from rich.text import Text

from agent_hardener.display.final import render_rich
from agent_hardener.display.final.render_rich import build_report_sections
from agent_hardener.display.final.report_tui import _ReportApp

pytestmark = pytest.mark.unit


def test_build_report_sections_keys_and_renderables() -> None:
    sections = build_report_sections([])
    assert list(sections) == ["Overview", "Attack", "Defense", "Validation", "Tables"]
    # Each section is a renderable that prints without error.
    console = Console(file=io.StringIO(), width=80)
    for renderable in sections.values():
        console.print(renderable)


def test_report_tui_composes_and_switches_tabs() -> None:
    async def _run() -> None:
        from textual.widgets import TabbedContent, TabPane  # noqa: PLC0415

        app = _ReportApp({"Overview": Text("ov"), "Attack": Text("atk")})
        async with app.run_test() as pilot:
            tabbed = app.query_one(TabbedContent)
            assert len(list(app.query(TabPane))) == 2
            assert tabbed.active == "tab-1"
            await pilot.press("right")
            assert tabbed.active == "tab-2"
            await pilot.press("q")

    asyncio.run(_run())


def test_print_final_summary_verbose_plain_when_not_rich(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Non-TTY verbose path writes the plain log and never reaches the TUI.
    monkeypatch.setattr(render_rich, "is_rich", lambda: False)
    render_rich.print_final_summary([], verbose=True)
    assert "Agent Hardener final log" in capsys.readouterr().out
