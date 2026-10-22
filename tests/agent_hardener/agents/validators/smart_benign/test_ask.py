# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the interviewer's ``ask`` node: the interactive prompt loop and its non-interactive guards."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.nodes import ask as ak
from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.schemas import (
    InterviewerQuestion,
    InterviewOption,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.state import InterviewerState
from agent_hardener.events import EventType

if TYPE_CHECKING:
    from collections.abc import Callable


def _option(description: str, *, recommended: bool = False) -> InterviewOption:
    return InterviewOption(label=description[:3], description=description, recommended=recommended)


def _question(text: str | None, gap: str = "gap", *options: InterviewOption) -> InterviewerQuestion:
    return InterviewerQuestion(question=text, gap=gap, options=list(options))


def _state(*questions: InterviewerQuestion, interactive: bool = True) -> InterviewerState:
    return InterviewerState(composed_questions=list(questions), interactive=interactive)


class _Prompt:
    """A questionary prompt whose ``ask()`` returns a scripted answer."""

    def __init__(self, answer: str | None) -> None:
        self.answer = answer

    def ask(self) -> str | None:
        return self.answer


class _Questionary:
    """Scripted ``questionary.select``/``text`` replacement that records how it was driven."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, selects: list[str | None], texts: list[str | None]) -> None:
        self._selects, self._texts = list(selects), list(texts)
        self.select_calls: list[dict[str, Any]] = []
        self.text_prompts: list[str] = []
        monkeypatch.setattr(ak.questionary, "select", self._select)
        monkeypatch.setattr(ak.questionary, "text", self._text)

    def _select(self, message: str, **kwargs: Any) -> _Prompt:
        self.select_calls.append({"message": message, **kwargs})
        return _Prompt(self._selects.pop(0))

    def _text(self, message: str, **_kwargs: Any) -> _Prompt:
        self.text_prompts.append(message)
        return _Prompt(self._texts.pop(0))


# --- single question with options --------------------------------------------------------------------------


def test_picking_an_option_returns_its_description(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _Questionary(monkeypatch, selects=["Customers"], texts=[])
    options = [_option("Employees"), _option("Customers", recommended=True)]

    assert ak._sync_ask_with_options("Who uses it?", options) == "Customers"

    (call,) = prompts.select_calls
    assert call["default"] == "Customers"  # the recommended option is pre-selected
    titles = [choice.title for choice in call["choices"]]
    employees, customers, other = titles
    assert "Employees" in employees
    assert "Customers" in customers
    assert customers.strip() != "Customers"  # only the recommended option carries a marker
    assert employees.strip() == "Employees"
    assert "Other" in other  # always last


def test_the_first_option_is_the_default_when_none_is_recommended(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _Questionary(monkeypatch, selects=["A"], texts=[])

    ak._sync_ask_with_options("Q?", [_option("A"), _option("B")])

    assert prompts.select_calls[0]["default"] == "A"


def test_other_prompts_for_free_text(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _Questionary(monkeypatch, selects=[ak._SENTINEL_OTHER], texts=["Auditors"])

    assert ak._sync_ask_with_options("Who uses it?", [_option("A")]) == "Auditors"
    assert prompts.text_prompts == ["  Your answer:"]


@pytest.mark.parametrize("typed", [None, ""], ids=["cancelled", "blank"])
def test_other_without_text_yields_no_answer(monkeypatch: pytest.MonkeyPatch, typed: str | None) -> None:
    _Questionary(monkeypatch, selects=[ak._SENTINEL_OTHER], texts=[typed])

    assert ak._sync_ask_with_options("Q?", [_option("A")]) is None


def test_ctrl_c_on_the_picker_yields_no_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    _Questionary(monkeypatch, selects=[None], texts=[])

    assert ak._sync_ask_with_options("Q?", [_option("A")]) is None


# --- batches -----------------------------------------------------------------------------------------------


def _typed(monkeypatch: pytest.MonkeyPatch, *lines: str) -> list[str]:
    remaining = list(lines)
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return remaining.pop(0)

    monkeypatch.setattr("builtins.input", fake_input)
    return prompts


def test_batch_collects_gap_question_answer_triples_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _Questionary(monkeypatch, selects=["Customers"], texts=[])
    prompts = _typed(monkeypatch, "  internal only  ")
    questions = [
        _question("Who uses it?", "personas", _option("Customers")),
        _question("Any limits?", "scope"),
    ]

    assert ak._sync_ask_batch(questions) == [
        ("personas", "Who uses it?", "Customers"),
        ("scope", "Any limits?", "internal only"),
    ]
    assert prompts == ["  > "]


def test_batch_skips_questions_the_llm_left_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _typed(monkeypatch, "yes")

    assert ak._sync_ask_batch([_question(None, "skipped"), _question("Ready?", "ready")]) == [
        ("ready", "Ready?", "yes")
    ]
    assert len(prompts) == 1


def test_batch_keeps_what_it_has_when_the_user_cancels_midway(monkeypatch: pytest.MonkeyPatch) -> None:
    _Questionary(monkeypatch, selects=["A", None, "C"], texts=[])
    questions = [_question(f"Q{i}", f"g{i}", _option("x")) for i in range(3)]

    assert ak._sync_ask_batch(questions) == [("g0", "Q0", "A")]


def test_a_blank_free_text_answer_ends_the_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    _typed(monkeypatch, "", "never asked")

    assert ak._sync_ask_batch([_question("Q1"), _question("Q2")]) == []


# --- the node ----------------------------------------------------------------------------------------------


class _Events:
    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.seen: list[str] = []
        monkeypatch.setattr(ak, "emit_event", lambda event, _payload: self.seen.append(str(event)))


def _tty(monkeypatch: pytest.MonkeyPatch, *, is_tty: bool = True) -> None:
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: is_tty))


def _batch(monkeypatch: pytest.MonkeyPatch, behaviour: Callable[[list[InterviewerQuestion]], Any]) -> None:
    monkeypatch.setattr(ak, "_sync_ask_batch", behaviour)


def test_a_non_interactive_run_never_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    events = _Events(monkeypatch)
    _batch(monkeypatch, lambda _q: pytest.fail("must not prompt"))

    delta = asyncio.run(ak.ask(_state(_question("Q?", "g"), interactive=False)))

    assert delta == {"source_note": "interviewer skipped (non-interactive)"}
    assert events.seen == []


def test_an_empty_batch_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    _batch(monkeypatch, lambda _q: pytest.fail("must not prompt"))

    delta = asyncio.run(ak.ask(_state(_question(None, "g"))))

    assert delta == {"source_note": "interviewer skipped (no questions to ask)"}


def test_a_missing_tty_is_reported_as_an_error_instead_of_hanging(monkeypatch: pytest.MonkeyPatch) -> None:
    _tty(monkeypatch, is_tty=False)
    events = _Events(monkeypatch)
    _batch(monkeypatch, lambda _q: pytest.fail("must not prompt"))

    delta = asyncio.run(ak.ask(_state(_question("Q?", "g"))))

    assert delta["source_note"] == "interviewer skipped (no TTY)"
    assert delta["errors"] == ["interviewer: interactive=True but stdin is not a TTY"]
    assert events.seen == []


def test_collected_answers_are_returned_inside_start_and_complete_events(monkeypatch: pytest.MonkeyPatch) -> None:
    _tty(monkeypatch)
    events = _Events(monkeypatch)
    answers = [("g1", "Q1", "A1"), ("g2", "Q2", "A2")]

    def batch(_questions: list[InterviewerQuestion]) -> list[tuple[str, str, str]]:
        events.seen.append("prompting")
        return answers

    _batch(monkeypatch, batch)

    delta = asyncio.run(ak.ask(_state(_question("Q1", "g1"), _question("Q2", "g2"))))

    assert delta == {"collected_answers": answers, "source_note": "collected 2 answer(s)"}
    assert events.seen == [str(EventType.INTERVIEW_STARTED), "prompting", str(EventType.INTERVIEW_COMPLETED)]


def test_the_whole_composed_list_goes_to_the_batch_including_blank_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    _tty(monkeypatch)
    _Events(monkeypatch)
    received: list[list[InterviewerQuestion]] = []

    def batch(questions: list[InterviewerQuestion]) -> list[tuple[str, str, str]]:
        received.append(questions)
        return [("g", "Q", "A")]

    _batch(monkeypatch, batch)
    mixed = [_question(None, "skip"), _question("Q", "g")]

    asyncio.run(ak.ask(_state(*mixed)))

    assert received == [mixed]  # the whole composed list goes through; the batch skips blanks itself


def test_no_answers_is_reported_but_not_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _tty(monkeypatch)
    events = _Events(monkeypatch)
    _batch(monkeypatch, lambda _q: [])

    delta = asyncio.run(ak.ask(_state(_question("Q?", "g"))))

    assert delta == {"source_note": "interviewer: no answers collected"}
    assert events.seen[-1] == str(EventType.INTERVIEW_COMPLETED)


@pytest.mark.parametrize("abort", [EOFError, KeyboardInterrupt])
def test_an_aborted_interview_still_resumes_the_display(
    monkeypatch: pytest.MonkeyPatch, abort: type[BaseException]
) -> None:
    _tty(monkeypatch)
    events = _Events(monkeypatch)

    def interrupted(_questions: list[InterviewerQuestion]) -> list[tuple[str, str, str]]:
        raise abort

    _batch(monkeypatch, interrupted)

    delta = asyncio.run(ak.ask(_state(_question("Q?", "g"))))

    assert delta == {"source_note": f"interview aborted ({abort.__name__})"}
    assert events.seen == [str(EventType.INTERVIEW_STARTED), str(EventType.INTERVIEW_COMPLETED)]
