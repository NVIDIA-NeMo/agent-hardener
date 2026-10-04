# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured, UI-ready diff of a defender's previous → current YAML.

The defenders phase rewrites victim YAMLs (agent workflow / OpenShell policy). This module
turns the ``before``/``after`` text both defenders already hold into a single typed contract
that is persisted as JSON and rendered identically by the CLI today and a future UI. It is
pure (no ``rich``); rendering lives in :mod:`agent_hardener.final_log.render_diff`.
"""

from __future__ import annotations

import difflib
import re
from typing import Any, Literal

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

DiffLineKind = Literal["context", "add", "remove"]

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class DiffLine(AgentHardenerModel):
    """One line of a hunk, with its line number on each side (``None`` where it does not exist)."""

    kind: DiffLineKind
    old_lineno: int | None = None
    new_lineno: int | None = None
    text: str = ""


class DiffHunk(AgentHardenerModel):
    """One ``@@`` change block: a contiguous run of context/added/removed lines."""

    header: str
    old_start: int = 0
    old_count: int = 0
    new_start: int = 0
    new_count: int = 0
    lines: list[DiffLine] = Field(default_factory=list)


class YamlDiff(AgentHardenerModel):
    """A previous → current YAML diff: the serializable contract the CLI and UI both render."""

    from_label: str = "current"
    to_label: str = "candidate"
    hunks: list[DiffHunk] = Field(default_factory=list)
    added: int = 0
    removed: int = 0

    @property
    def is_empty(self) -> bool:
        """Whether the two sides are identical (no hunks)."""
        return not self.hunks

    def to_json(self) -> dict[str, Any]:
        """Return the plain JSON-able mapping (for ``write_json`` / the UI)."""
        return self.model_dump(mode="json")


def build_yaml_diff(
    before: str | None,
    after: str | None,
    *,
    from_label: str = "current",
    to_label: str = "candidate",
    context: int = 3,
) -> YamlDiff:
    """Build a :class:`YamlDiff` from two YAML texts.

    Returns an empty diff when either side is missing or the two sides are identical.

    Args:
        before: Previous YAML text (the live config before the defender ran).
        after: Current YAML text (the defender's candidate).
        from_label: Label for the previous side (the ``---`` file).
        to_label: Label for the current side (the ``+++`` file).
        context: Number of unchanged context lines around each change.

    Returns:
        The structured diff.
    """
    diff = YamlDiff(from_label=from_label, to_label=to_label)
    if before is None or after is None or before == after:
        return diff
    unified = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=from_label,
        tofile=to_label,
        lineterm="",
        n=context,
    )
    return _parse_lines(list(unified), from_label=from_label, to_label=to_label)


def parse_unified_diff(text: str, *, from_label: str = "current", to_label: str = "candidate") -> YamlDiff:
    """Build a :class:`YamlDiff` from already-written unified-diff text (a ``.diff`` file)."""
    return _parse_lines(text.splitlines(), from_label=from_label, to_label=to_label)


def _parse_lines(lines: list[str], *, from_label: str, to_label: str) -> YamlDiff:
    diff = YamlDiff(from_label=from_label, to_label=to_label)
    hunk: DiffHunk | None = None
    old_lineno = new_lineno = 0
    for raw in lines:
        if raw.startswith(("--- ", "+++ ")) or raw.startswith("diff "):
            continue
        match = _HUNK_HEADER.match(raw)
        if match:
            old_start, old_count, new_start, new_count = (
                int(match.group(1)),
                int(match.group(2) or 1),
                int(match.group(3)),
                int(match.group(4) or 1),
            )
            hunk = DiffHunk(
                header=raw,
                old_start=old_start,
                old_count=old_count,
                new_start=new_start,
                new_count=new_count,
            )
            diff.hunks.append(hunk)
            old_lineno, new_lineno = old_start, new_start
            continue
        if hunk is None:
            continue
        prefix, body = raw[:1], raw[1:]
        if prefix == "+":
            hunk.lines.append(DiffLine(kind="add", new_lineno=new_lineno, text=body))
            new_lineno += 1
            diff.added += 1
        elif prefix == "-":
            hunk.lines.append(DiffLine(kind="remove", old_lineno=old_lineno, text=body))
            old_lineno += 1
            diff.removed += 1
        elif prefix in (" ", ""):
            hunk.lines.append(DiffLine(kind="context", old_lineno=old_lineno, new_lineno=new_lineno, text=body))
            old_lineno += 1
            new_lineno += 1
    return diff
