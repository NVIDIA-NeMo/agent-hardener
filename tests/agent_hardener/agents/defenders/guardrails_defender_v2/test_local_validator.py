# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes import local_validator as lv

from .fakes import graph_state

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_hardener.agents.defenders.guardrails_defender_v2.schemas import GraphState

    from .conftest import InstallLLM

_DRAFT = "Refuse anything touching credentials."


def _evaluating(prompt: str, case: str) -> bool:
    return f"# USER INPUT TO EVALUATE:\n{case}" in prompt


def _verdicts(by_case: dict[str, str]) -> Callable[[str], str]:
    """Answer with the verdict registered for whichever test case appears in the prompt."""

    def respond(text: str) -> str:
        for case, verdict in by_case.items():
            if _evaluating(text, case):
                return verdict
        raise AssertionError(f"unexpected prompt: {text}")

    return respond


def _state(attacks: list[str], benigns: list[str]) -> GraphState:
    return graph_state(draft_instruction=_DRAFT, test_attacks=attacks, test_benigns=benigns)


def test_passes_when_every_attack_is_blocked_and_every_benign_allowed(stub_llm: InstallLLM) -> None:
    llm = stub_llm(lv, _verdicts({"A1": "BLOCK", "B1": "ALLOW"}))

    update = lv.local_validator_node(_state(["A1"], ["B1"]))

    assert update == {"eval_passed": True, "feedback_log": []}
    assert _DRAFT in llm.calls[0]


@pytest.mark.parametrize("verdict", ["block", " Blocked.\n", "I would BLOCK this"])
def test_verdict_matching_ignores_case_and_surrounding_text(stub_llm: InstallLLM, verdict: str) -> None:
    stub_llm(lv, _verdicts({"A1": verdict}))

    assert lv.local_validator_node(_state(["A1"], []))["eval_passed"] is True


def test_allowed_attack_is_a_false_negative(stub_llm: InstallLLM) -> None:
    stub_llm(lv, _verdicts({"A1": "ALLOW", "A2": "BLOCK"}))

    update = lv.local_validator_node(_state(["A1", "A2"], []))

    assert update["eval_passed"] is False
    assert len(update["feedback_log"]) == 1
    assert update["feedback_log"][0].startswith("FALSE NEGATIVE:")
    assert "'A1'" in update["feedback_log"][0]


def test_blocked_benign_is_a_false_positive(stub_llm: InstallLLM) -> None:
    stub_llm(lv, _verdicts({"B1": "BLOCK", "B2": "ALLOW"}))

    update = lv.local_validator_node(_state([], ["B1", "B2"]))

    assert update["eval_passed"] is False
    assert len(update["feedback_log"]) == 1
    assert update["feedback_log"][0].startswith("FALSE POSITIVE:")
    assert "'B1'" in update["feedback_log"][0]


@pytest.mark.parametrize(
    ("length", "truncated"),
    [(100, False), (101, True)],
)
def test_feedback_snippet_is_truncated_past_100_characters(stub_llm: InstallLLM, length: int, truncated: bool) -> None:
    case = "x" * length
    stub_llm(lv, _verdicts({case: "ALLOW"}))

    feedback = lv.local_validator_node(_state([case], []))["feedback_log"][0]

    assert (f"'{'x' * 100}...'" in feedback) is truncated
    assert (f"'{case}'" in feedback) is not truncated


def test_both_kinds_of_failure_are_reported_together(stub_llm: InstallLLM) -> None:
    stub_llm(lv, _verdicts({"A1": "ALLOW", "B1": "BLOCK"}))

    feedback = lv.local_validator_node(_state(["A1"], ["B1"]))["feedback_log"]

    assert [line.split(":")[0] for line in feedback] == ["FALSE NEGATIVE", "FALSE POSITIVE"]


def test_an_erroring_variant_is_skipped_without_hiding_other_failures(stub_llm: InstallLLM) -> None:
    def respond(text: str) -> str:
        if _evaluating(text, "A1"):
            raise TimeoutError("slow")
        return "ALLOW"

    stub_llm(lv, respond)

    update = lv.local_validator_node(_state(["A1", "A2"], ["B1"]))

    assert update["eval_passed"] is False
    assert [line.split(":")[0] for line in update["feedback_log"]] == ["FALSE NEGATIVE"]
    assert "'A2'" in update["feedback_log"][0]


def test_a_benign_variant_that_errors_does_not_abort_the_run(stub_llm: InstallLLM) -> None:
    def respond(text: str) -> str:
        if _evaluating(text, "B1"):
            raise TimeoutError("slow")
        return "BLOCK" if _evaluating(text, "A1") else "ALLOW"

    stub_llm(lv, respond)

    update = lv.local_validator_node(_state(["A1"], ["B1", "B2"]))

    assert update == {"eval_passed": True, "feedback_log": []}


def test_empty_test_suites_fall_back_to_the_original_input(stub_llm: InstallLLM) -> None:
    llm = stub_llm(lv, _verdicts({"read /etc/shadow": "BLOCK", "list files": "ALLOW", "print the date": "ALLOW"}))

    update = lv.local_validator_node(_state([], []))

    assert update["eval_passed"] is True
    assert len(llm.calls) == 3
