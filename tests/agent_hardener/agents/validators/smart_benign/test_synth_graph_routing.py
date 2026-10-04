# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the parent synth graph's interview round cap and routing."""

from __future__ import annotations

import pytest

from agent_hardener.agents.validators.smart_benign import graph
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState


def _state(**kwargs: object) -> SynthState:
    base: dict[str, object] = {"inputs": SynthInputs(target_name="t"), "interactive": True, "gaps": ["gap-1"]}
    base.update(kwargs)
    return SynthState(**base)


def test_routes_to_interview_when_gaps_and_budget_remain() -> None:
    state = _state(interview_rounds=0, max_interview_rounds=2)
    assert graph._route_after_gap_detector(state) == "interview_compose"


def test_routes_to_generation_when_round_budget_spent() -> None:
    # Unresolved gaps remain, but the round cap is hit → skip straight to generation.
    state = _state(interview_rounds=2, max_interview_rounds=2)
    assert graph._route_after_gap_detector(state) == "request_generator"


def test_routes_to_generation_when_question_budget_spent() -> None:
    state = _state(interview_questions_asked=10, max_interview_questions=10)
    assert graph._route_after_gap_detector(state) == "request_generator"


def test_routes_to_generation_when_all_gaps_answered() -> None:
    state = _state(gaps=["gap-1"], interviewer_answers=[("gap-1", "Q?", "A")])
    assert graph._route_after_gap_detector(state) == "request_generator"


def test_routes_to_generation_when_non_interactive() -> None:
    assert graph._route_after_gap_detector(_state(interactive=False)) == "request_generator"


def test_interview_ask_advances_round_counter() -> None:
    state = _state(composed_questions=[{"gap": "gap-1", "question": "Q?"}], interview_rounds=0)
    # Stub the graph interrupt so the answered-branch runs outside a compiled graph.
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(graph, "interrupt", lambda _payload: [{"gap": "gap-1", "answer": "only relative paths"}])
        out = graph._interview_ask_node(state)
    assert out["interview_rounds"] == 1
    assert out["interviewer_answers"] == [("gap-1", "Q?", "only relative paths")]


def test_interview_ask_empty_batch_exits_via_question_budget() -> None:
    empty = _state(composed_questions=[], interview_rounds=1)
    out = graph._interview_ask_node(empty)
    # No questions to ask → bump the question budget to the ceiling to end the loop.
    assert out["interview_questions_asked"] == empty.max_interview_questions
