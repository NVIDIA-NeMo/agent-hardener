# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a :class:`~agent_hardener.yaml_diff.YamlDiff` as a GitHub-style diff.

Two presentations over one structured diff: ``diff_renderable`` (colored ``rich`` for a TTY)
and ``diff_plain_lines`` (unified-diff text for non-TTY / the plain log). ``diff_stat`` is the
compact ``+N -M`` headline used in the default report. ``load_yaml_diff_from_patch`` resolves a
defender policy patch to its diff so both the CLI and a future UI read the same source.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from rich.console import Group
from rich.text import Text

from agent_hardener.final_log import load_json
from agent_hardener.yaml_diff import DiffLine, YamlDiff, build_yaml_diff, parse_unified_diff

# GitHub-style colors: added green, removed red, hunk header cyan, context dim.
_ADD_STYLE = "green"
_REMOVE_STYLE = "red"
_HUNK_STYLE = "cyan"
_GUTTER_STYLE = "dim"


def diff_stat(diff: YamlDiff) -> Text:
    """Compact ``+N -M`` change stat (green adds, red removes); ``no change`` when empty."""
    if diff.is_empty:
        return Text("no change", style="dim")
    text = Text()
    text.append(f"+{diff.added}", style=_ADD_STYLE)
    text.append(" ")
    text.append(f"-{diff.removed}", style=_REMOVE_STYLE)
    return text


def _capped_diff_items(diff: YamlDiff, max_lines: int) -> tuple[list[tuple[str, object]], bool, int]:
    """Walk the diff's hunks up to ``max_lines`` body lines.

    Returns flat ``("header", hunk_header) | ("line", DiffLine)`` items, whether it was truncated, and
    how many body lines were dropped — so the rich and plain presenters share one traversal.
    """
    items: list[tuple[str, object]] = []
    rendered = 0
    truncated = False
    for hunk in diff.hunks:
        if rendered >= max_lines:
            truncated = True
            break
        items.append(("header", hunk.header))
        for line in hunk.lines:
            if rendered >= max_lines:
                truncated = True
                break
            items.append(("line", line))
            rendered += 1
    remaining = sum(len(hunk.lines) for hunk in diff.hunks) - rendered
    return items, truncated, remaining


def diff_renderable(diff: YamlDiff, *, max_lines: int = 200, more_hint: str = "") -> Group:
    """Build a colored GitHub-style renderable for ``diff`` (truncated past ``max_lines``)."""
    parts: list[Text] = [_diff_title(diff)]
    if diff.is_empty:
        parts.append(Text("  (no changes)", style="dim"))
        return Group(*parts)
    items, truncated, remaining = _capped_diff_items(diff, max_lines)
    for kind, value in items:
        parts.append(Text(str(value), style=_HUNK_STYLE) if kind == "header" else _diff_line(cast("DiffLine", value)))
    if truncated:
        hint = f" (see {more_hint})" if more_hint else ""
        parts.append(Text(f"  … {remaining} more diff line(s){hint}", style="dim"))
    return Group(*parts)


def diff_plain_lines(diff: YamlDiff, *, max_lines: int = 200) -> list[str]:
    """Render ``diff`` as plain unified-diff text lines (non-TTY / the plain log)."""
    if diff.is_empty:
        return [f"{diff.from_label} -> {diff.to_label}: no changes"]
    lines = [f"--- {diff.from_label}", f"+++ {diff.to_label}"]
    items, truncated, remaining = _capped_diff_items(diff, max_lines)
    for kind, value in items:
        if kind == "header":
            lines.append(str(value))
        else:
            line = cast("DiffLine", value)
            lines.append(f"{_PREFIX[line.kind]}{line.text}")
    if truncated:
        lines.append(f"... {remaining} more diff line(s)")
    return lines


def load_yaml_diff_from_patch(patch: dict) -> tuple[str, YamlDiff] | None:
    """Resolve a defender policy patch to ``(label, YamlDiff)``, or ``None`` when nothing changed.

    Prefers the persisted structured ``*_diff_json_path``; falls back to the before/after
    snapshot YAMLs, then to parsing the raw ``.diff`` text.
    """
    if patch.get("changed") is False:
        return None
    label = _patch_label(patch)
    diff = _diff_from_json(patch) or _diff_from_snapshots(patch) or _diff_from_raw(patch)
    if diff is None or diff.is_empty:
        return None
    return label, diff


# --- internals -------------------------------------------------------------

_PREFIX = {"context": " ", "add": "+", "remove": "-"}
_GUTTER_WIDTH = 4

_JSON_KEYS = ("workflow_diff_json_path", "policy_diff_json_path")
_BEFORE_KEYS = ("current_workflow_path", "current_policy_path")
_AFTER_KEYS = ("candidate_workflow_path", "candidate_policy_path")
_RAW_KEYS = ("workflow_diff_path", "diff_path")
_LABEL_KEYS = ("target_workflow_path", "candidate_workflow_path", "candidate_policy_path")


def _diff_title(diff: YamlDiff) -> Text:
    text = Text()
    text.append(f"{diff.from_label} → {diff.to_label}  ", style="bold")
    text.append_text(diff_stat(diff))
    return text


def _diff_line(line: DiffLine) -> Text:
    gutter = f"{_fmt_no(line.old_lineno)}{_fmt_no(line.new_lineno)}"
    style = {"add": _ADD_STYLE, "remove": _REMOVE_STYLE}.get(line.kind, "")
    text = Text()
    text.append(gutter, style=_GUTTER_STYLE)
    text.append(f" {_PREFIX[line.kind]} ", style=style)
    text.append(line.text, style=style)
    return text


def _fmt_no(value: int | None) -> str:
    return f"{value:>{_GUTTER_WIDTH}}" if value is not None else " " * _GUTTER_WIDTH


def _first_path(patch: dict, keys: tuple[str, ...]) -> Path | None:
    for key in keys:
        value = patch.get(key)
        if value:
            return Path(str(value))
    return None


def _patch_label(patch: dict) -> str:
    path = _first_path(patch, _LABEL_KEYS)
    return path.name if path is not None else str(patch.get("type", "yaml"))


def _diff_from_json(patch: dict) -> YamlDiff | None:
    path = _first_path(patch, _JSON_KEYS)
    if path is None or not path.exists():
        return None
    data = load_json(path)
    return YamlDiff.model_validate(data) if data else None


def _diff_from_snapshots(patch: dict) -> YamlDiff | None:
    before_path = _first_path(patch, _BEFORE_KEYS)
    after_path = _first_path(patch, _AFTER_KEYS)
    before = _read_text(before_path)
    after = _read_text(after_path)
    if before is None or after is None:
        return None
    return build_yaml_diff(before, after)


def _diff_from_raw(patch: dict) -> YamlDiff | None:
    text = _read_text(_first_path(patch, _RAW_KEYS))
    return parse_unified_diff(text) if text else None


def _read_text(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None
