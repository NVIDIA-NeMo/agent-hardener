# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The validators' shared trace step, and reports that count trace relabels without changing the rest."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.validators.smart_benign import validator as benign
from agent_hardener.agents.validators.smart_benign.replay_judge.reporting import format_finding, stats_key
from agent_hardener.agents.validators.smart_benign.replay_judge.traces import benign_outcome
from agent_hardener.agents.validators.trace_step import TraceWindow, final_status, open_trace_window
from agent_hardener.models import AgentConfig
from agent_hardener.runtime.run_context import RunContext
from agent_hardener.traces import AgentTrace
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)


def _trace(trace_id: str, at: int) -> AgentTrace:
    start = T0 + timedelta(seconds=at)
    return AgentTrace(trace_id=trace_id, input="p", output="", started_at=start, ended_at=start + timedelta(seconds=1))


class BrokenProvider(FakeTraceProvider):
    def get_traces(self) -> list[AgentTrace]:
        raise RuntimeError("sandbox gone")


def test_no_ctx_or_no_provider_means_no_window() -> None:
    assert asyncio.run(open_trace_window(None)) is None
    assert asyncio.run(open_trace_window(RunContext())) is None


def test_a_failing_anchor_read_falls_back_to_no_window() -> None:
    assert asyncio.run(open_trace_window(RunContext(trace_provider=BrokenProvider()))) is None


def test_the_window_keeps_only_traffic_after_it_opened() -> None:
    provider = FakeTraceProvider(batches=[[_trace("old", 0)], [_trace("new", 10)]])
    window = asyncio.run(open_trace_window(RunContext(trace_provider=provider)))
    assert isinstance(window, TraceWindow)
    assert window.anchor == T0 + timedelta(seconds=1)
    assert [trace.trace_id for trace in asyncio.run(window.traces())] == ["new"]


def test_final_status_prefers_the_trace_relabel() -> None:
    assert final_status({"verdict": {"status": "refused"}}) == "refused"
    assert final_status({"verdict": {"status": "refused"}, "trace_verdict": {"status": "blocked_guardrail"}}) == (
        "blocked_guardrail"
    )


def _row(status: str, trace_status: str | None = None) -> dict:
    row = {
        "index": 0,
        "tool": "read",
        "label": "benign",
        "payload_excerpt": "x",
        "verdict": {"status": status, "confidence": 0.9, "reasoning": "r"},
    }
    if trace_status:
        row["trace_verdict"] = {"status": trace_status}
    return row


def _report(tmp_path: Path, rows: list[dict]):
    agent = AgentConfig(name="smart-benign", role="validator", config={"kind": "benign"})
    return benign._build_report(agent, tmp_path, rows, started=0.0)


def test_benign_report_without_traces_is_unchanged(tmp_path: Path) -> None:
    report = _report(tmp_path, [_row("complied"), _row("refused")])
    assert report.ok is False
    assert {key for key in report.metadata if key.endswith("_count")} == {
        "complied_count",
        "refused_count",
        "error_count",
        "not_counted_count",
    }
    assert report.metadata["per_tool"] == {"read": {"complied": 1, "refused": 1, "errors": 0}}
    assert report.findings[1].startswith("REFUSED ")


def test_benign_report_counts_relabels_and_only_our_blocks_fail_it(tmp_path: Path) -> None:
    neutral = _report(tmp_path, [_row("complied"), _row("refused", "agent_refused")])
    assert neutral.ok is True
    assert neutral.metadata["refused_count"] == 0
    assert neutral.metadata["agent_refused_count"] == 1
    assert neutral.metadata["per_tool"]["read"]["agent_refused"] == 1

    ours = _report(tmp_path, [_row("refused", "blocked_network")])
    assert ours.ok is False
    assert format_finding(_row("refused", "blocked_network")).startswith("BLOCKED_NETWORK ")
    assert stats_key("error") == "errors"


@pytest.mark.parametrize(
    ("status", "trace_verdict", "expected"),
    [
        ("complied", None, "served"),
        ("error", None, "error"),
        ("refused", None, "refused"),  # no trace to explain it
        ("refused", {"status": "blocked_guardrail"}, "refused"),
        ("refused", {"status": "blocked_network"}, "refused"),
        ("refused", {"status": "agent_refused"}, "not_counted"),
        ("refused", {"status": "tool_failed", "args_match_request": True}, "served"),
        ("refused", {"status": "tool_failed", "args_match_request": False}, "refused"),
        ("refused", {"status": "tool_failed"}, "refused"),  # no judge, or it could not answer
    ],
)
def test_how_each_benign_status_counts(status: str, trace_verdict: dict | None, expected: str) -> None:
    row = _row(status)
    if trace_verdict:
        row["trace_verdict"] = trace_verdict
    assert benign_outcome(row) == expected
