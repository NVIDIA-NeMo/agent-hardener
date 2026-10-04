# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive tabbed TUI for the ``--verbose`` final report.

Hosts the existing Rich section renderables (from :func:`render_rich.build_report_sections`) in a
Textual ``TabbedContent`` so an operator can jump between Overview / Attack / Defense / Validation /
Tables instead of scrolling one long paged document. Each tab is a scrollable pane. Launched only on
an interactive TTY; callers fall back to the pager if Textual is unavailable or the app fails to run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual.app import App
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.widgets import Footer, Header, Static, TabbedContent, TabPane, Tabs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rich.console import RenderableType
    from textual.app import ComposeResult


class _ReportApp(App[None]):
    """Tabbed viewer for the final report sections."""

    TITLE = "Agent Hardener — Final Report"
    CSS = "VerticalScroll { padding: 0 1; }"
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "quit", "Quit"),
        Binding("left", "prev_tab", "Prev"),
        Binding("right", "next_tab", "Next"),
    ]

    def __init__(self, sections: Mapping[str, RenderableType]) -> None:
        super().__init__()
        self._sections = dict(sections)

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            for title, renderable in self._sections.items():
                with TabPane(title):
                    yield VerticalScroll(Static(renderable))
        yield Footer()

    def action_next_tab(self) -> None:
        self.query_one(Tabs).action_next_tab()

    def action_prev_tab(self) -> None:
        self.query_one(Tabs).action_previous_tab()


def run_report_tui(sections: Mapping[str, RenderableType]) -> None:
    """Run the tabbed report viewer (blocking; takes over the terminal, restores it on exit)."""
    _ReportApp(sections).run()
