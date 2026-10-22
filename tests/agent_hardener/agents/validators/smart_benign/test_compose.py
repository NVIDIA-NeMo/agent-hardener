# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the interviewer's ``compose`` node: gap filtering, the question-batch LLM call, and failures."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest

from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.nodes import compose as cp
from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.schemas import (
    InterviewBatch,
    InterviewerQuestion,
    InterviewOption,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.interviewer.state import InterviewerState
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable


def _state(*gaps: str, **overrides: Any) -> InterviewerState:
    base = {"gaps": list(gaps), "synth_model": "m", "synth_base_url": "http://llm.test", "synth_api_key": "k"}
    return InterviewerState(**{**base, **overrides})


def _patch(monkeypatch: pytest.MonkeyPatch, model: Any) -> Any:
    monkeypatch.setattr(cp, "build_chat_model", lambda **_kw: model)
    return model


def _forbid_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cp, "build_chat_model", lambda **_kw: pytest.fail("must not call the LLM"))


def _batch(*gaps: str) -> InterviewBatch:
    return InterviewBatch(questions=[InterviewerQuestion(question=f"About {g}?", gap=g) for g in gaps])


def test_no_gaps_skips_the_round(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_llm(monkeypatch)

    assert asyncio.run(cp.compose(_state())) == {
        "composed_questions": [],
        "source_note": "interviewer skipped (no gaps)",
    }


def test_gaps_already_answered_are_not_asked_again(monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_llm(monkeypatch)
    state = _state("scope", "personas", prior_answers=[("scope", "Q?", "A"), ("personas", "Q2?", "A2")])

    delta = asyncio.run(cp.compose(state))

    assert delta == {
        "composed_questions": [],
        "source_note": "interviewer skipped (all gaps addressed in prior Q&A)",
    }


def test_only_unasked_gaps_reach_the_prompt(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    model = _patch(monkeypatch, fake_model(_batch("personas")))
    state = _state("gap-answered", "gap-open", prior_answers=[("gap-answered", "Q?", "A")])

    delta = asyncio.run(cp.compose(state))

    user = model.messages[1].content
    assert "- gap-open" in user
    assert "- gap-answered" not in user  # still visible as prior Q&A, but not listed as a current gap
    assert [q.gap for q in cast("list[InterviewerQuestion]", delta["composed_questions"])] == ["personas"]


def test_the_prompt_carries_the_known_agent_context(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = _patch(monkeypatch, fake_model(_batch("scope")))
    state = _state(
        "scope",
        remaining_budget=4,
        system_role="billing assistant",
        tool_summaries=["refund_tool — issues refunds"],
        out_of_scope=["tax advice"],
        known_personas=["Dana: accountant"],
        prior_answers=[("", "Seeded Q?", "Seeded A")],
    )

    asyncio.run(cp.compose(state))

    system, user = model.messages
    assert system.content
    for expected in ("scope", "billing assistant", "refund_tool", "tax advice", "Dana: accountant", "Seeded Q?"):
        assert expected in user.content


def test_the_llm_questions_are_returned_untouched(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    question = InterviewerQuestion(
        question="Who uses it?",
        gap="personas",
        options=[InterviewOption(label="A", description="Admins", recommended=True)],
    )
    _patch(monkeypatch, fake_model(InterviewBatch(questions=[question])))

    delta = asyncio.run(cp.compose(_state("personas")))

    assert delta == {"composed_questions": [question]}


def test_an_empty_llm_batch_is_a_normal_outcome(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(InterviewBatch()))

    assert asyncio.run(cp.compose(_state("scope"))) == {"composed_questions": []}


def test_an_llm_failure_skips_the_round_and_reports_the_error(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=ValueError("schema mismatch")))

    delta = asyncio.run(cp.compose(_state("scope")))

    assert delta["composed_questions"] == []
    assert delta["errors"] == ["interviewer.compose: schema mismatch"]
    assert "schema mismatch" in str(delta["source_note"])


def test_a_rate_limit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(monkeypatch, fake_model(raises=RateLimitError("slow down", source="llm")))

    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(cp.compose(_state("scope")))


def test_a_provider_rate_limit_message_is_promoted(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=RuntimeError("rate limit hit")))

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(cp.compose(_state("scope")))

    assert exc_info.value.source == "smart_benign:interviewer"


@pytest.mark.parametrize(
    ("gaps", "prior", "expected"),
    [
        (["a", "b"], [], ["a", "b"]),
        (["a", "b"], [("a", "q", "x")], ["b"]),
        (["a"], [("", "seeded", "x")], ["a"]),  # a blank gap tag never matches a real gap
        (["a"], [("A", "q", "x")], ["a"]),  # exact, case-sensitive match
        (["a", "a"], [], ["a", "a"]),
    ],
    ids=["none-asked", "one-asked", "blank-tag", "case-sensitive", "duplicates-kept"],
)
def test_unasked_gaps(gaps: list[str], prior: list[tuple[str, str, str]], expected: list[str]) -> None:
    assert cp._unasked_gaps(gaps, prior) == expected
