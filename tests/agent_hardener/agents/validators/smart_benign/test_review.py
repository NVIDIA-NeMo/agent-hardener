# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the interactive benign-suite review (pure parts only; the TTY loop is manual)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.validators.smart_benign import review
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.review import _edit_tool, format_suite
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.nodes.write import _write_requests_csv
from agent_hardener.agents.validators.smart_benign.validator import load_requests

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _req(tool: str, payload: str, label: str = "benign") -> GeneratedRequest:
    return GeneratedRequest(tool=tool, payload=payload, label=label)  # type: ignore[arg-type]


def _script(monkeypatch: pytest.MonkeyPatch, *, selects: list[object], texts: list[str]) -> None:
    """Feed scripted answers to questionary.select/.text so the edit loop runs without a TTY."""
    sels, txts = iter(selects), iter(texts)

    class _Ask:
        def __init__(self, value: object) -> None:
            self._value = value

        def ask(self) -> object:
            return self._value

    monkeypatch.setattr(review.questionary, "select", lambda *_a, **_k: _Ask(next(sels)))
    monkeypatch.setattr(review.questionary, "text", lambda *_a, **_k: _Ask(next(txts)))


def test_format_suite_groups_by_tool() -> None:
    lines = format_suite([_req("bash", "ls"), _req("bash", "pwd", "borderline_benign"), _req("py", "print(1)")])
    assert lines[0] == "Benign suite — 3 request(s):"
    assert "  bash (2):" in lines
    assert "    [benign] ls" in lines
    assert "    [borderline_benign] pwd" in lines
    assert "  py (1):" in lines


def test_edited_suite_round_trips_through_requests_csv(tmp_path: Path) -> None:
    # An edit (replace one payload) persists via the same writer/loader the review uses.
    reqs = [_req("bash", "ls"), _req("py", "print(1)")]
    edited = [r.model_copy(update={"payload": "ls -la"}) if r.tool == "bash" else r for r in reqs]
    _write_requests_csv(tmp_path, edited)
    reloaded = load_requests(tmp_path / "requests.csv")
    assert [(r.tool, r.payload) for r in reloaded] == [("bash", "ls -la"), ("py", "print(1)")]


def test_edit_tool_cancel_and_back_do_not_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: questionary.Choice coerces value=None to the title, so sentinels must be explicit.
    # 'cancel' must not be treated as a row index ("list indices must be integers, not str").
    reqs = [_req("bash", "ls"), _req("bash", "pwd")]
    _script(monkeypatch, selects=["delete", review._CANCEL, review._BACK], texts=[])
    assert _edit_tool(reqs, "bash") == reqs  # cancel → no-op, back → return unchanged


def test_edit_tool_edits_and_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    reqs = [_req("bash", "ls"), _req("bash", "pwd")]
    _script(monkeypatch, selects=["edit", 0, "delete", 1, review._BACK], texts=["ls -la"])
    result = _edit_tool(reqs, "bash")
    assert [r.payload for r in result] == ["ls -la"]  # row 0 edited, then row 1 deleted
