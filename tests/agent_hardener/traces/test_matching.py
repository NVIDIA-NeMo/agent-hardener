# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Picking the trace that belongs to an attack hit or a benign request, and windowing by time."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agent_hardener.traces import AgentTrace, find_attack_trace, find_benign_traces, trace_clock, traces_since

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _trace(trace_id: str, input: str, output: str = "", at: int = 0) -> AgentTrace:  # noqa: A002
    start = T0 + timedelta(seconds=at)
    return AgentTrace(
        trace_id=trace_id, input=input, output=output, started_at=start, ended_at=start + timedelta(seconds=1)
    )


def test_same_prompt_sent_twice_picks_the_attempt_with_the_matching_response() -> None:
    traces = [_trace("a", "steal it", "I refuse.", at=0), _trace("b", "steal it", "Sent the data to x", at=5)]
    assert find_attack_trace(traces, "steal  it", "Sent the data").trace_id == "b"


def test_ties_are_broken_by_the_hit_time_then_by_recency() -> None:
    traces = [_trace("a", "p", "ok", at=0), _trace("b", "p", "ok", at=10)]
    assert find_attack_trace(traces, "p", "ok", at=T0 + timedelta(seconds=1)).trace_id == "a"
    assert find_attack_trace(traces, "p", "ok").trace_id == "b"


def test_a_reshaped_answer_still_matches_by_input() -> None:
    """The transport may wrap the agent's answer, so an output mismatch keeps the input matches."""
    traces = [_trace("a", "p", "raw answer")]
    assert find_attack_trace(traces, "p", '{"reply": "raw answer"}').trace_id == "a"


def test_no_input_match_returns_none() -> None:
    assert find_attack_trace([_trace("a", "other")], "p", "x") is None
    assert find_attack_trace([], "p", "x") is None


def test_benign_returns_every_run_oldest_first_and_since_drops_older_runs() -> None:
    traces = [_trace("b", "hi", at=10), _trace("a", "hi", at=0), _trace("c", "bye", at=20)]
    assert [t.trace_id for t in find_benign_traces(traces, "hi")] == ["a", "b"]
    assert [t.trace_id for t in find_benign_traces(traces, "hi", since=T0 + timedelta(seconds=5))] == ["b"]


def test_claim_once_gives_identical_prompts_distinct_traces() -> None:
    traces = [_trace("a", "p", "ok", at=0), _trace("b", "p", "ok", at=1)]
    claimed: set[str] = set()
    first = find_attack_trace(traces, "p", "ok", claimed=claimed)
    second = find_attack_trace(traces, "p", "ok", claimed=claimed)
    assert {first.trace_id, second.trace_id} == {"a", "b"}
    assert find_attack_trace(traces, "p", "ok", claimed=claimed) is None

    benign_claimed: set[str] = set()
    assert len(find_benign_traces(traces, "p", claimed=benign_claimed)) == 2
    assert find_benign_traces(traces, "p", claimed=benign_claimed) == []


def test_window_keeps_only_traffic_after_the_anchor() -> None:
    stale = [_trace("old", "p", at=0)]
    anchor = trace_clock(stale)
    assert anchor == T0 + timedelta(seconds=1)
    fresh = _trace("new", "p", at=5)
    window = traces_since([*stale, fresh], anchor)
    assert [t.trace_id for t in window] == ["new"]
    assert find_attack_trace(window, "p", "").trace_id == "new"


def test_an_empty_list_gives_no_anchor_and_keeps_everything() -> None:
    assert trace_clock([]) is None
    traces = [_trace("a", "p")]
    assert traces_since(traces, None) == traces
