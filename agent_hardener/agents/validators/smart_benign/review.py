# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive review of the synthesized benign suite before an ``agent-hardener run`` uses it.

The operator sees the generated request payloads (grouped by tool) and confirms them; on decline
they edit, per tool, the exact prompts that will be replayed. Edits are persisted to ``requests.csv``
(the input hash is left untouched, so the edited suite survives across runs). TTY-only — the runner
gates this behind ``_is_interactive()``. Ctrl+C at any prompt proceeds with the current suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import questionary

from .models import GeneratedRequest
from .subgraphs.profile_writer.nodes.write import _write_requests_csv
from .validator import REQUESTS_CSV, load_requests

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# questionary.Choice coerces a None value to its title, so sentinels must be explicit non-None values.
_DONE = "\x00done"
_BACK = "\x00back"
_CANCEL = "\x00cancel"


def format_suite(requests: list[GeneratedRequest]) -> list[str]:
    """Render the suite as display lines, grouped by tool (each row's label + payload)."""
    lines = [f"Benign suite — {len(requests)} request(s):"]
    for tool in dict.fromkeys(r.tool for r in requests):
        rows = [r for r in requests if r.tool == tool]
        lines.append(f"  {tool} ({len(rows)}):")
        lines.extend(f"    [{r.label}] {r.payload}" for r in rows)
    return lines


def review_benign_suite(artifact_dir: Path, *, on_output: Callable[[str], None]) -> None:
    """Show + confirm the suite; on decline edit per-tool and persist back to ``requests.csv``."""
    csv_path = artifact_dir / REQUESTS_CSV
    requests = load_requests(csv_path)
    if not requests:
        return
    while True:
        for line in format_suite(requests):
            on_output(line)
        if questionary.confirm("Use this benign suite?", default=True).ask() is not False:
            return  # approved (or Ctrl+C → don't block the run)
        requests = _edit_loop(requests)
        _write_requests_csv(artifact_dir, requests)
        on_output(f"updated benign suite: {len(requests)} request(s) → {csv_path}")


def _edit_loop(requests: list[GeneratedRequest]) -> list[GeneratedRequest]:
    """Per-tool edit menu: pick a tool to edit, or finish."""
    while True:
        tools = list(dict.fromkeys(r.tool for r in requests))
        tool = questionary.select(
            "Edit which tool's requests?",
            choices=[*tools, questionary.Choice(title="Done — use this suite", value=_DONE)],
        ).ask()
        if tool is None or tool == _DONE:  # Done or Ctrl+C
            return requests
        requests = _edit_tool(requests, tool)


def _edit_tool(requests: list[GeneratedRequest], tool: str) -> list[GeneratedRequest]:
    """Edit / add / delete the request payloads for a single tool (verbatim, no LLM)."""
    while True:
        rows = [r for r in requests if r.tool == tool]
        action = questionary.select(
            f"{tool} — {len(rows)} request(s)",
            choices=["edit", "add", "delete", questionary.Choice(title="back", value=_BACK)],
        ).ask()
        if action is None or action == _BACK:  # back or Ctrl+C
            return requests
        if action == "add":
            payload = (questionary.text("New request payload:").ask() or "").strip()
            if payload:
                requests = [*requests, GeneratedRequest(tool=tool, payload=payload, label="benign")]
            continue
        if not rows:
            continue
        idx = questionary.select(
            f"{action} which request?",
            choices=[questionary.Choice(title=f"[{r.label}] {r.payload}", value=i) for i, r in enumerate(rows)]
            + [questionary.Choice(title="cancel", value=_CANCEL)],
        ).ask()
        if idx is None or idx == _CANCEL:
            continue
        target = rows[idx]
        if action == "edit":
            payload = (questionary.text("Edit payload:", default=target.payload).ask() or "").strip()
            if payload:
                requests = [r.model_copy(update={"payload": payload}) if r is target else r for r in requests]
        else:  # delete
            requests = [r for r in requests if r is not target]
