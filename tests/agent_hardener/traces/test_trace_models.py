# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trace objects cross the agent boundary as JSON, so they must round-trip exactly."""

from __future__ import annotations

from datetime import UTC, datetime

from agent_hardener.traces import AgentTrace, LlmCall, ToolCall, render_trace
from agent_hardener.traces.testing import FakeTraceProvider

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _trace() -> AgentTrace:
    return AgentTrace(
        trace_id="root-1",
        input="Use the transfer_funds tool",
        output="Done.",
        started_at=T0,
        ended_at=T0,
        steps=[
            LlmCall(messages=[{"role": "user", "content": "Use the transfer_funds tool"}], output=""),
            ToolCall(
                name="transfer_funds",
                args={"amount": 900},
                result="agent-hardener[pii]: refused transfer_funds (safety score 0.91 >= 0.50)",
                ok=False,
                outcome="guardrail_refused",
                rail="pii",
            ),
            LlmCall(output="Done."),
        ],
        guardrail_latched=True,
    )


def test_trace_round_trips_through_json_with_its_step_kinds() -> None:
    trace = _trace()
    loaded = AgentTrace.model_validate_json(trace.model_dump_json())
    assert loaded == trace
    assert [type(step) for step in loaded.steps] == [LlmCall, ToolCall, LlmCall]


def test_tool_and_llm_calls_are_split_in_order() -> None:
    trace = _trace()
    assert [call.name for call in trace.tool_calls] == ["transfer_funds"]
    assert [call.output for call in trace.llm_calls] == ["", "Done."]


def test_render_shows_each_step_and_who_refused() -> None:
    text = render_trace(_trace(), limit=40)
    assert text.splitlines()[0] == "USER: Use the transfer_funds tool"
    assert "TOOL transfer_funds" in text
    assert "[guardrail_refused by pii]" in text
    assert "latched" in text
    assert text.splitlines()[-1] == "ANSWER: Done."
    assert all(len(line) < 200 for line in text.splitlines())


def test_fake_provider_counts_calls_and_releases_batches() -> None:
    first, second = _trace(), _trace().model_copy(update={"trace_id": "root-2"})
    provider = FakeTraceProvider(batches=[[first], [second]])
    assert provider.get_traces() == [first]
    assert provider.get_traces() == [first, second]
    assert provider.get_traces() == [first, second]
    assert provider.calls == 3
    assert FakeTraceProvider().get_traces() == []
