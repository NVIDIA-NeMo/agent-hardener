# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Flow tests for the interactive suite review: approve, decline-and-edit, persistence, and Ctrl+C handling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign import review
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.nodes.write import _write_requests_csv
from agent_hardener.agents.validators.smart_benign.validator import REQUESTS_CSV, load_requests

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def _req(tool: str, payload: str, label: Any = "benign") -> GeneratedRequest:
    return GeneratedRequest(tool=tool, payload=payload, label=label)


class _Ask:
    def __init__(self, value: object) -> None:
        self._value = value

    def ask(self) -> object:
        return self._value


class _Script:
    """Scripted questionary answers; a test that over-asks fails loudly instead of hanging."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        confirms: list[object] | None = None,
        selects: list[object] | None = None,
        texts: list[object] | None = None,
    ) -> None:
        self.confirms, self.selects, self.texts = list(confirms or []), list(selects or []), list(texts or [])
        self.select_choices: list[list[Any]] = []
        self.text_defaults: list[Any] = []
        monkeypatch.setattr(review.questionary, "confirm", lambda *_a, **_k: _Ask(self.confirms.pop(0)))
        monkeypatch.setattr(review.questionary, "select", self._select)
        monkeypatch.setattr(review.questionary, "text", self._text)

    def _select(self, _message: str, **kwargs: Any) -> _Ask:
        self.select_choices.append(kwargs["choices"])
        return _Ask(self.selects.pop(0))

    def _text(self, _message: str, **kwargs: Any) -> _Ask:
        self.text_defaults.append(kwargs.get("default"))
        return _Ask(self.texts.pop(0))


def _seed(tmp_path: Path, *requests: GeneratedRequest) -> None:
    _write_requests_csv(tmp_path, list(requests))


def _payloads(tmp_path: Path) -> list[tuple[str, str]]:
    return [(r.tool, r.payload) for r in load_requests(tmp_path / REQUESTS_CSV)]


def test_format_suite_for_an_empty_suite_is_just_the_header() -> None:
    assert review.format_suite([]) == ["Benign suite — 0 request(s):"]


def test_format_suite_keeps_first_seen_tool_order_when_rows_interleave() -> None:
    lines = review.format_suite([_req("b", "1"), _req("a", "2"), _req("b", "3")])

    assert [line for line in lines if line.startswith("  ") and not line.startswith("    ")] == ["  b (2):", "  a (1):"]


def test_an_empty_suite_returns_without_prompting_or_printing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path)
    _Script(monkeypatch)  # any prompt would pop from an empty list and raise
    out: list[str] = []

    review.review_benign_suite(tmp_path, on_output=out.append)

    assert out == []


def test_a_missing_csv_surfaces_the_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        review.review_benign_suite(tmp_path, on_output=lambda _line: None)


def test_approving_prints_the_suite_and_leaves_the_file_alone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path, _req("bash", "ls"))
    before = (tmp_path / REQUESTS_CSV).read_bytes()
    _Script(monkeypatch, confirms=[True])
    out: list[str] = []

    review.review_benign_suite(tmp_path, on_output=out.append)

    assert out == review.format_suite([_req("bash", "ls")])
    assert (tmp_path / REQUESTS_CSV).read_bytes() == before


def test_ctrl_c_on_the_confirm_prompt_proceeds_with_the_current_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, _req("bash", "ls"))
    _Script(monkeypatch, confirms=[None], selects=[])  # no edit prompt may follow

    review.review_benign_suite(tmp_path, on_output=lambda _line: None)

    assert _payloads(tmp_path) == [("bash", "ls")]


def test_declining_then_editing_persists_and_re_shows_the_updated_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, _req("bash", "ls"), _req("py", "print(1)"))
    script = _Script(
        monkeypatch,
        confirms=[False, True],
        # tool picker -> bash; action -> edit; row -> 0; back; tool picker -> done
        selects=["bash", "edit", 0, review._BACK, review._DONE],
        texts=["ls -la"],
    )
    out: list[str] = []

    review.review_benign_suite(tmp_path, on_output=out.append)

    assert _payloads(tmp_path) == [("bash", "ls -la"), ("py", "print(1)")]
    assert script.text_defaults == ["ls"]  # the edit prompt is pre-filled with the current payload
    assert f"updated benign suite: 2 request(s) → {tmp_path / REQUESTS_CSV}" in out
    assert "    [benign] ls -la" in out  # the second round shows the edited row


def test_the_tool_picker_offers_each_tool_once_plus_done(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path, _req("bash", "a"), _req("py", "b"), _req("bash", "c"))
    script = _Script(monkeypatch, confirms=[False, True], selects=[review._DONE])

    review.review_benign_suite(tmp_path, on_output=lambda _line: None)

    first = script.select_choices[0]
    assert first[:2] == ["bash", "py"]
    assert first[2].value == review._DONE


def test_declining_with_a_cancelled_picker_keeps_the_suite_and_asks_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(tmp_path, _req("bash", "ls"))
    _Script(monkeypatch, confirms=[False, True], selects=[None])

    review.review_benign_suite(tmp_path, on_output=lambda _line: None)

    assert _payloads(tmp_path) == [("bash", "ls")]


def test_adding_a_request_appends_a_benign_row_for_that_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    _Script(monkeypatch, selects=["add", review._BACK], texts=["  brand new  "])

    result = review._edit_tool([_req("bash", "ls"), _req("py", "x")], "bash")

    assert [(r.tool, r.payload, r.label) for r in result] == [
        ("bash", "ls", "benign"),
        ("py", "x", "benign"),
        ("bash", "brand new", "benign"),
    ]


@pytest.mark.parametrize("typed", ["", "   ", None], ids=["empty", "blank", "ctrl-c"])
def test_adding_a_blank_payload_is_ignored(monkeypatch: pytest.MonkeyPatch, typed: str | None) -> None:
    _Script(monkeypatch, selects=["add", review._BACK], texts=[typed])
    requests = [_req("bash", "ls")]

    assert review._edit_tool(requests, "bash") == requests


@pytest.mark.parametrize("typed", ["", "   ", None], ids=["empty", "blank", "ctrl-c"])
def test_editing_to_a_blank_payload_keeps_the_original(monkeypatch: pytest.MonkeyPatch, typed: str | None) -> None:
    _Script(monkeypatch, selects=["edit", 0, review._BACK], texts=[typed])

    result = review._edit_tool([_req("bash", "ls")], "bash")

    assert [r.payload for r in result] == ["ls"]


def test_editing_preserves_the_label_and_only_touches_the_chosen_row(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = [_req("bash", "ls", "borderline_benign"), _req("bash", "ls"), _req("py", "ls")]
    _Script(monkeypatch, selects=["edit", 0, review._BACK], texts=["pwd"])

    result = review._edit_tool(requests, "bash")

    assert [(r.tool, r.payload, r.label) for r in result] == [
        ("bash", "pwd", "borderline_benign"),
        ("bash", "ls", "benign"),
        ("py", "ls", "benign"),
    ]


def test_deleting_removes_only_the_chosen_row_even_among_identical_ones(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = [_req("bash", "ls"), _req("bash", "ls", "borderline_benign")]
    _Script(monkeypatch, selects=["delete", 1, review._BACK])

    result = review._edit_tool(requests, "bash")

    assert [(r.payload, r.label) for r in result] == [("ls", "benign")]


def test_the_row_picker_is_scoped_to_the_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, selects=["delete", review._CANCEL, review._BACK])

    review._edit_tool([_req("bash", "ls"), _req("py", "print(1)")], "py")

    titles = [c.title for c in script.select_choices[1]]
    assert titles == ["[benign] print(1)", "cancel"]


def test_edit_or_delete_on_a_tool_with_no_rows_loops_back_to_the_menu(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, selects=["delete", "edit", review._BACK])

    assert review._edit_tool([_req("bash", "ls")], "ghost") == [_req("bash", "ls")]
    assert len(script.select_choices) == 3  # never reached the row picker


def test_ctrl_c_on_the_action_menu_returns_the_work_so_far(monkeypatch: pytest.MonkeyPatch) -> None:
    _Script(monkeypatch, selects=["add", None], texts=["kept"])

    result = review._edit_tool([_req("bash", "ls")], "bash")

    assert [r.payload for r in result] == ["ls", "kept"]


def test_deleting_every_row_leaves_a_header_only_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed(tmp_path, _req("bash", "ls"))
    _Script(
        monkeypatch,
        confirms=[False, True],
        selects=["bash", "delete", 0, review._BACK, review._DONE],
    )
    out: list[str] = []

    review.review_benign_suite(tmp_path, on_output=out.append)

    assert load_requests(tmp_path / REQUESTS_CSV) == []
    assert f"updated benign suite: 0 request(s) → {tmp_path / REQUESTS_CSV}" in out
