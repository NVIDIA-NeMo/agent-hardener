# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the terminal interview answer providers (display/interview.py).

Covers the pure recommendation logic and the batch loop; the questionary/stdin edges are driven by
monkeypatching ``_ask_with_options`` and ``builtins.input`` so no TTY is needed.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from agent_hardener.display import interview

if TYPE_CHECKING:
    from typing import Any

    import pytest


def _opts() -> list[dict[str, Any]]:
    return [{"description": "first"}, {"description": "second", "recommended": True}]


def test_recommended_prefers_flagged_then_first() -> None:
    assert interview._recommended(_opts()) == "second"  # the flagged one
    assert interview._recommended([{"description": "only"}, {"description": "other"}]) == "only"  # else first


def test_default_answer_uses_recommended_and_skips_free_text() -> None:
    picked = interview._default_answer({"gap": "g", "question": "q?", "options": _opts()})
    assert picked == {"gap": "g", "question": "q?", "answer": "second"}
    assert interview._default_answer({"question": "free text?"}) is None  # no options → unresolved


def test_auto_answer_provider_accepts_recommended_and_omits_optionless() -> None:
    questions = [
        {"gap": "g1", "question": "a?", "options": _opts()},
        {"gap": "g2", "question": "free?"},  # no options → skipped
        {"gap": "g3", "question": "b?", "options": [{"description": "x"}]},
    ]
    answers = asyncio.run(interview.auto_answer_provider(questions))
    assert answers == [
        {"gap": "g1", "question": "a?", "answer": "second"},
        {"gap": "g3", "question": "b?", "answer": "x"},
    ]


def test_ask_batch_collects_options_and_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(interview, "_ask_with_options", lambda _q, _o: "chosen")
    monkeypatch.setattr("builtins.input", lambda _prompt="": "typed")

    collected = interview._ask_batch(
        [
            {"gap": "g1", "question": "pick?", "options": _opts()},
            {"gap": "g2", "question": "type?"},  # free text
            {"question": ""},  # blank question skipped
        ]
    )
    assert collected == [
        {"gap": "g1", "question": "pick?", "answer": "chosen"},
        {"gap": "g2", "question": "type?", "answer": "typed"},
    ]


def test_ask_batch_stops_on_ctrl_c(monkeypatch: pytest.MonkeyPatch) -> None:
    # _ask_with_options returns None on Ctrl+C — the batch must stop with what it has.
    monkeypatch.setattr(interview, "_ask_with_options", lambda _q, _o: None)
    collected = interview._ask_batch([{"gap": "g1", "question": "pick?", "options": _opts()}])
    assert collected == []
