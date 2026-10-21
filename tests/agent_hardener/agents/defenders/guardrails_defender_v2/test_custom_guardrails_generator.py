# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes import custom_guardrails_generator as gen
from agent_hardener.models import ValidationFeedback
from agent_hardener.relay_plugin.config import DEFAULT_GUARDRAIL_THRESHOLD

from .fakes import FakeLLM, graph_state

if TYPE_CHECKING:
    from .conftest import InstallLLM


class _Script:
    """Hands out suggestor drafts and grader verdicts in order, telling them apart by system prompt."""

    def __init__(self, drafts: list[str], grades: list[str]) -> None:
        self._drafts = iter(drafts)
        self._grades = iter(grades)

    def __call__(self, text: str) -> str:
        return next(self._grades if gen.GRADER_SYS_PROMPT in text else self._drafts)


def _grade(score: int) -> str:
    return f"[SCORE]: {score}\n[STRENGTHS]: s\n[WEAKNESSES]: w{score}\n[REQUIRED FIXES]: f"


def _run(stub_llm: InstallLLM, drafts: list[str], grades: list[str], **state_fields: Any) -> tuple[str, FakeLLM]:
    llm = stub_llm(gen, _Script(drafts, grades))
    draft = gen.custom_guardrails_generator_node(graph_state(**state_fields))["draft_instruction"]
    return draft, llm


def _suggestor_prompts(llm: FakeLLM) -> list[str]:
    return llm.calls[0::2]


def _grader_prompts(llm: FakeLLM) -> list[str]:
    return llm.calls[1::2]


def test_a_high_first_score_ends_the_loop_after_one_round(stub_llm: InstallLLM) -> None:
    draft, llm = _run(stub_llm, ["draft one"], [_grade(9)])

    assert draft == "draft one"
    assert len(llm.calls) == 2


@pytest.mark.parametrize(
    ("grade", "rounds"),
    [(_grade(10), 1), (_grade(9), 1), (_grade(8), 3), ("no verdict here", 3), ("[SCORE]: nine", 3)],
)
def test_loop_stops_at_score_nine_and_otherwise_runs_three_rounds(
    stub_llm: InstallLLM, grade: str, rounds: int
) -> None:
    _, llm = _run(stub_llm, ["d1", "d2", "d3"], [grade] * 3)

    assert len(llm.calls) == 2 * rounds


def test_the_last_draft_is_kept_when_the_score_never_passes(stub_llm: InstallLLM) -> None:
    draft, _ = _run(stub_llm, ["d1", "d2", "d3"], [_grade(4)] * 3)

    assert draft == "d3"


def test_passing_on_the_second_round_keeps_that_round_s_draft(stub_llm: InstallLLM) -> None:
    draft, llm = _run(stub_llm, ["d1", "d2", "d3"], [_grade(5), _grade(9)])

    assert draft == "d2"
    assert len(llm.calls) == 4


def test_each_refinement_sees_the_previous_draft_and_its_grader_feedback(stub_llm: InstallLLM) -> None:
    _, llm = _run(stub_llm, ["d1", "d2", "d3"], [_grade(5), _grade(6), _grade(7)])

    second, third = _suggestor_prompts(llm)[1:]
    assert "# YOUR PREVIOUS DRAFT:\nd1" in second
    assert "w5" in second
    assert "# YOUR PREVIOUS DRAFT:\nd2" in third
    assert "w6" in third
    assert "w5" not in third


def test_first_prompt_describes_the_attack(stub_llm: InstallLLM) -> None:
    _, llm = _run(stub_llm, ["d1"], [_grade(9)])

    first = _suggestor_prompts(llm)[0]
    assert "# TARGET TOOL:\nbash" in first
    assert "read /etc/shadow" in first
    assert "root:x:0:0" in first
    assert gen.JUDGE_SYSTEM_PROMPT in first


def test_grader_is_given_the_draft_the_attack_and_the_threshold(stub_llm: InstallLLM) -> None:
    _, llm = _run(stub_llm, ["the draft"], [_grade(9)], vulnerability_analysis="REPORT-TEXT")

    grader = _grader_prompts(llm)[0]
    assert "the draft" in grader
    assert "REPORT-TEXT" in grader
    assert "read /etc/shadow" in grader
    assert f"{DEFAULT_GUARDRAIL_THRESHOLD} or higher" in grader


@pytest.mark.parametrize("analysis", ["REPORT-TEXT", ""])
def test_threat_intelligence_is_added_only_when_there_is_an_analysis(stub_llm: InstallLLM, analysis: str) -> None:
    _, llm = _run(stub_llm, ["d1"], [_grade(9)], vulnerability_analysis=analysis)

    first = _suggestor_prompts(llm)[0]
    assert ("# THREAT INTELLIGENCE REPORT\nREPORT-TEXT" in first) is bool(analysis)


def test_no_validation_feedback_section_without_prior_feedback(stub_llm: InstallLLM) -> None:
    _, llm = _run(stub_llm, ["d1"], [_grade(9)])

    assert "PREVIOUS ATTEMPT FEEDBACK" not in _suggestor_prompts(llm)[0]


@pytest.mark.parametrize(
    ("feedback", "present", "absent"),
    [
        (
            ValidationFeedback(false_negatives=["FN-1", "FN-2"]),
            ["FALSE NEGATIVES", "  - FN-1", "  - FN-2"],
            ["FALSE POSITIVES", "PREVIOUS POLICY"],
        ),
        (
            ValidationFeedback(false_positives=["FP-1"]),
            ["FALSE POSITIVES", "  - FP-1"],
            ["FALSE NEGATIVES", "PREVIOUS POLICY"],
        ),
        (
            ValidationFeedback(previous_policy_yaml="old: yaml"),
            ["PREVIOUS POLICY (that failed):\nold: yaml"],
            ["FALSE NEGATIVES", "FALSE POSITIVES"],
        ),
        (
            ValidationFeedback(false_negatives=["FN-1"], false_positives=["FP-1"], previous_policy_yaml="old: yaml"),
            ["FALSE NEGATIVES", "FALSE POSITIVES", "PREVIOUS POLICY"],
            [],
        ),
    ],
)
def test_validation_feedback_lists_only_the_failures_that_occurred(
    stub_llm: InstallLLM, feedback: ValidationFeedback, present: list[str], absent: list[str]
) -> None:
    _, llm = _run(stub_llm, ["d1"], [_grade(9)], input_overrides={"feedback": feedback})

    first = _suggestor_prompts(llm)[0]
    assert "# PREVIOUS ATTEMPT FEEDBACK" in first
    assert first.rstrip().endswith("Address all issues above in your revised guardrail.")
    assert all(fragment in first for fragment in present)
    assert not any(fragment in first for fragment in absent)


def test_validation_feedback_only_goes_into_the_first_prompt(stub_llm: InstallLLM) -> None:
    feedback = ValidationFeedback(false_negatives=["FN-1"])
    _, llm = _run(stub_llm, ["d1", "d2"], [_grade(5), _grade(9)], input_overrides={"feedback": feedback})

    assert "FN-1" not in _suggestor_prompts(llm)[1]


def test_outer_loop_feedback_turns_the_first_round_into_a_refinement(stub_llm: InstallLLM) -> None:
    _, llm = _run(stub_llm, ["d1"], [_grade(9)], feedback_log=["missed variant X", "blocked benign Y"])

    first = _suggestor_prompts(llm)[0]
    assert "Previous Graph Level Feedback to incorporate:\n- missed variant X\n- blocked benign Y" in first
    assert "AUDITOR FEEDBACK" in first
    assert "# TARGET TOOL" not in first


@pytest.mark.parametrize(
    ("raw", "cleaned"),
    [
        ("  plain text \n", "plain text"),
        ('"quoted"', "quoted"),
        ("`inline`", "inline"),
    ],
)
def test_draft_is_stripped_of_whitespace_and_wrapping_quotes(stub_llm: InstallLLM, raw: str, cleaned: str) -> None:
    draft, _ = _run(stub_llm, [raw], [_grade(9)])

    assert draft == cleaned


@pytest.mark.parametrize("raw", ["```text\nScore 1.0 for attacks.\n```", "```\nScore 1.0 for attacks.\n```"])
def test_fenced_code_block_is_unwrapped(stub_llm: InstallLLM, raw: str) -> None:
    draft, _ = _run(stub_llm, [raw], [_grade(9)])

    assert draft == "Score 1.0 for attacks."
