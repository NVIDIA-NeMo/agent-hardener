# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the guardrail decision itself.

No Relay here on purpose: this is the logic worth testing hard, since an over-broad guardrail breaks
the agent and an under-broad one lets the attack through, and neither needs a Relay runtime to show.
"""

from __future__ import annotations

from agent_hardener.relay_plugin.config import Guardrail
from agent_hardener.relay_plugin.policy import decide


class _Judge:
    """A judge returning a fixed score, recording whether it was consulted."""

    def __init__(self, score: float = 1.0) -> None:
        self.score_value = score
        self.calls: list[tuple[str, str]] = []

    def score(self, instructions: str, tool_name: str, args: str) -> float:
        del instructions
        self.calls.append((tool_name, args))
        return self.score_value


class _RaisingJudge:
    def score(self, instructions: str, tool_name: str, args: str) -> float:
        raise RuntimeError("inference endpoint unreachable")


def _rail(**kwargs: object) -> Guardrail:
    base = {"name": "custom_guardrail_1", "target_tool": "transfer_funds", "system_instructions": "block exfil"}
    return Guardrail.model_validate({**base, **kwargs})


def test_a_matching_call_over_the_threshold_is_refused() -> None:
    decision = decide(_rail(), "transfer_funds", {"amount": 9999}, _Judge(0.9))
    assert decision.refuse
    assert "custom_guardrail_1" in decision.reason  # traceable back to the rail in the run report
    assert "transfer_funds" in decision.reason


def test_a_matching_call_under_the_threshold_is_allowed() -> None:
    assert not decide(_rail(threshold=0.7), "transfer_funds", {"amount": 5}, _Judge(0.3)).refuse


def test_the_threshold_boundary_refuses() -> None:
    assert decide(_rail(threshold=0.7), "transfer_funds", {}, _Judge(0.7)).refuse


def test_a_namespaced_target_tool_still_guards_the_bare_call() -> None:
    """The defender writes ``target_tool`` from the attack transcript, namespace and all.

    Relay reports ``bash_executor``; the rail can say ``functions.bash_executor``. Compared
    exactly those differ, and the mismatch reads as "not my tool" — so every call sails past
    unjudged and the run reports a clean pass for a guardrail that never ran. Observed for real:
    17 guardrails, 168 tool calls, zero judge consultations.
    """
    judge = _Judge(0.9)
    decision = decide(
        _rail(target_tool="functions.bash_executor"), "bash_executor", {"command": "cat ~/.ssh/id_rsa"}, judge
    )
    assert decision.refuse
    assert judge.calls, "the judge must actually be consulted, not skipped as a non-matching tool"


def test_a_namespace_is_not_a_licence_to_match_any_suffix() -> None:
    """Only the known ``functions.`` namespace is stripped — ``a.b`` must not be guarded by ``b``."""
    judge = _Judge(1.0)
    assert not decide(_rail(target_tool="other.transfer_funds"), "transfer_funds", {}, judge).refuse
    assert judge.calls == []


def test_another_tool_is_allowed_without_consulting_the_judge() -> None:
    """The judge is an LLM call in the tool path; the agent's ordinary work must not pay for it."""
    judge = _Judge(1.0)
    assert not decide(_rail(), "read_file", {"path": "README.md"}, judge).refuse
    assert judge.calls == []


def test_a_failing_judge_allows_rather_than_refuses() -> None:
    """A judge that cannot be reached must not refuse everything on the tool it guards.

    Failing closed would turn a transient inference error into an agent that refuses every call,
    which reads in the report as a guardrail so effective it broke the agent.
    """
    assert not decide(_rail(), "transfer_funds", {"amount": 9999}, _RaisingJudge()).refuse


def test_odd_argument_shapes_still_reach_the_judge() -> None:
    """Relay hands the callback any JSON value; a tool taking a bare string is legal.

    The judge scores the tool call, so an unserialisable or non-object payload must render to
    *something* rather than raise — a guardrail that errors here would fail open on the very calls
    an attacker is most likely to craft.
    """
    for args in ({"when": object()}, "just a string", 42, None):
        judge = _Judge(0.0)
        assert not decide(_rail(), "transfer_funds", args, judge).refuse
        assert judge.calls
        assert judge.calls[-1][0] == "transfer_funds"
