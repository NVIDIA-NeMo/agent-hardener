# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tool facts and the two inspection tables (agents/validators/README.md)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_hardener.traces import AgentTrace, ToolCall
from agent_hardener.traces.inspect import (
    inspect_attack,
    inspect_benign_block,
    needs_refusal_check,
    tool_facts,
)

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _call(name: str, outcome: str = "ok", rail: str | None = None, result: str = "r") -> ToolCall:
    return ToolCall(name=name, result=result, ok=outcome == "ok", outcome=outcome, rail=rail)  # type: ignore[arg-type]


def _trace(*calls: ToolCall, latched: bool = False) -> AgentTrace:
    return AgentTrace(
        trace_id="t1", input="p", output="a", started_at=T0, ended_at=T0, steps=list(calls), guardrail_latched=latched
    )


# --- tool facts ----------------------------------------------------------------------------------


def test_facts_for_a_target_tool_that_ran() -> None:
    facts = tool_facts(_trace(_call("read"), _call("transfer_funds")), "transfer_funds")
    assert (facts.target_tool_called, facts.target_tool_outcome, facts.target_tool_calls) == (True, "ok", 1)
    assert facts.target_tool_ran
    assert facts.tools_called == ["read", "transfer_funds"]
    assert (facts.blocked_tool, facts.failed_tool, facts.trace_id) == (None, None, "t1")


@pytest.mark.parametrize(
    ("outcome", "blocked_by"),
    [("guardrail_refused", "guardrail"), ("network_denied", "network"), ("error", None)],
)
def test_facts_for_a_stopped_target_tool(outcome: str, blocked_by: str | None) -> None:
    facts = tool_facts(_trace(_call("send_email", outcome, rail="exfil")), "send_email")
    assert (facts.target_tool_outcome, facts.target_tool_ran, facts.blocked_by) == (outcome, False, blocked_by)


def test_facts_for_a_target_tool_never_called() -> None:
    facts = tool_facts(_trace(_call("read")), "send_email")
    assert (facts.target_tool_called, facts.target_tool_outcome, facts.target_tool_calls) == (False, "not_called", 0)


def test_the_worst_outcome_wins_but_one_good_run_still_counts_as_ran() -> None:
    facts = tool_facts(_trace(_call("x"), _call("x", "error"), _call("x", "guardrail_refused", "r1")), "x")
    assert (facts.target_tool_outcome, facts.target_tool_calls, facts.target_tool_ran) == ("guardrail_refused", 3, True)


def test_unknown_target_fills_only_the_trace_wide_facts() -> None:
    facts = tool_facts(_trace(_call("fetch", "network_denied"), _call("y", "error", result="boom")), None)
    assert (facts.target_tool, facts.target_tool_called) == (None, False)
    assert (facts.blocked_tool, facts.blocked_by, facts.failed_tool, facts.failed_excerpt) == (
        "fetch",
        "network",
        "y",
        "boom",
    )


def test_a_namespaced_tool_name_matches_its_target() -> None:
    assert tool_facts(_trace(_call("functions.bash_executor")), "bash_executor").target_tool_ran


# --- attack table (section 4) --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("calls", "latched", "refused", "expected"),
    [
        ([_call("x")], False, False, ("not_blocked", "x", None)),
        ([_call("x", "guardrail_refused", "r1")], False, False, ("blocked_guardrail", "x", "r1")),
        ([_call("y", "guardrail_refused", "r1")], False, False, ("blocked_guardrail", "y", "r1")),
        ([], True, False, ("blocked_guardrail", None, None)),
        ([_call("x", "network_denied")], False, False, ("blocked_network", "x", None)),
        ([_call("fetch", "network_denied")], False, False, ("blocked_network", "fetch", None)),
        ([_call("x", "error")], False, False, ("tool_failed", "x", None)),
        ([_call("y")], False, True, ("agent_refused", None, None)),
        ([_call("y")], False, False, ("tool_not_called", None, None)),
    ],
)
def test_attack_rows(calls: list[ToolCall], latched: bool, refused: bool, expected: tuple) -> None:
    result = inspect_attack(tool_facts(_trace(*calls, latched=latched), "x"), agent_refused=refused)
    assert (result.status, result.on_tool, result.rail) == expected


def test_a_latched_request_is_reported_as_latched() -> None:
    result = inspect_attack(tool_facts(_trace(_call("x", "guardrail_refused", "r1"), latched=True), "x"))
    assert result.latched


def test_the_refusal_judge_is_needed_only_when_nothing_explains_the_missing_call() -> None:
    assert needs_refusal_check(tool_facts(_trace(_call("y")), "x"))
    assert not needs_refusal_check(tool_facts(_trace(_call("x")), "x"))
    assert not needs_refusal_check(tool_facts(_trace(_call("y", "network_denied")), "x"))
    assert not needs_refusal_check(tool_facts(_trace(latched=True), "x"))


# --- benign table (section 5) --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("calls", "latched", "expected"),
    [
        ([_call("read"), _call("send", "guardrail_refused", "pii")], False, ("blocked_guardrail", "send", "pii")),
        ([], True, ("blocked_guardrail", None, None)),
        ([_call("fetch", "network_denied")], False, ("blocked_network", "fetch", None)),
        ([_call("read", "error", result="Timeout")], False, ("tool_failed", "read", None)),
        ([_call("read")], False, ("agent_refused", None, None)),
    ],
)
def test_benign_rows(calls: list[ToolCall], latched: bool, expected: tuple) -> None:
    result = inspect_benign_block(tool_facts(_trace(*calls, latched=latched), "read"))
    assert (result.status, result.blocked_tool, result.rail) == expected


def test_benign_reports_the_error_and_the_tools_that_ran() -> None:
    failed = inspect_benign_block(tool_facts(_trace(_call("read", "error", result="Timeout")), "read"))
    assert failed.error_excerpt == "Timeout"
    refused = inspect_benign_block(tool_facts(_trace(_call("read"), _call("list")), "read"))
    assert refused.tools_called == ["read", "list"]
