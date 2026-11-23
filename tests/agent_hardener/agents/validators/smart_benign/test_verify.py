# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-synth tool verification: which suite tools the victim's traces show it really has."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.smart_benign import verify
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.trace_step import TraceWindow
from agent_hardener.traces import AgentTrace, ToolCall
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    import pytest

T0 = datetime(2026, 11, 22, 12, 0, tzinfo=UTC)


def _req(tool: str, payload: str) -> GeneratedRequest:
    return GeneratedRequest(tool=tool, payload=payload, label="benign")


def _trace(payload: str, at: int, *tools: str, args: Any = None) -> AgentTrace:
    start = T0 + timedelta(seconds=at)
    calls = [ToolCall(name=t, args=args, result="ok", ok=True, outcome="ok") for t in tools]
    return AgentTrace(
        trace_id=f"t{at}", input=payload, output="", started_at=start, ended_at=start + timedelta(seconds=1),
        steps=list(calls),
    )  # fmt: skip


REQUESTS = [
    _req("read_balance", "my balance"),
    _req("send_email", "email my statement"),  # never called by itself, but by another row: verified
    _req("teleport", "teleport me home"),  # traced, never called anywhere: missing
    _req("summon_cat", "summon a cat"),  # no trace at all: unknown
    _req("read_balance", "balance of ACC-9"),
]
TRACES: list[AgentTrace | None] = [
    _trace("my balance", 1, "functions.read_balance"),
    _trace("email my statement", 2, "lookup_contact"),
    _trace("teleport me home", 3),
    None,
    _trace("balance of ACC-9", 5, "read_balance", "send_email", "lookup_contact", args={"to": "me"}),
]


def test_classify_tools_reads_each_status_off_the_traces() -> None:
    verdicts = {v.tool: v for v in verify.classify_tools(REQUESTS, TRACES)}

    assert {tool: v.status for tool, v in verdicts.items()} == {
        "read_balance": "verified",
        "send_email": "verified",
        "teleport": "missing",
        "summon_cat": "unknown",
    }
    balance = verdicts["read_balance"]
    assert (balance.rows, balance.traced_rows, balance.rows_calling_it, balance.calls_anywhere) == (2, 2, 2, 2)
    email = verdicts["send_email"]
    assert (email.rows_calling_it, email.calls_anywhere) == (0, 1)


def test_unlisted_tools_collects_calls_of_tools_the_profile_does_not_list() -> None:
    observed = verify.unlisted_tools(["read_balance", "send_email"], REQUESTS, TRACES, max_calls=1)

    assert list(observed) == ["lookup_contact"]
    (call,) = observed["lookup_contact"]  # capped at max_calls
    assert (call.tool, call.prompt) == ("lookup_contact", "email my statement")


def test_unlisted_tools_keys_by_the_bare_name() -> None:
    traces: list[AgentTrace | None] = [_trace("my balance", 1, "functions.lookup_contact")]
    observed = verify.unlisted_tools([], REQUESTS[:1], traces, max_calls=5)
    assert list(observed) == ["lookup_contact"]


def _fake_run_all(statuses: list[str]) -> Any:
    async def run_all(requests: list[GeneratedRequest], _cfg: Any) -> list[dict[str, Any]]:
        return [
            {"index": i, "tool": r.tool, "payload_excerpt": r.payload, "verdict": {"status": s}}
            for i, (r, s) in enumerate(zip(requests, statuses, strict=True), start=1)
        ]

    return run_all


def test_verify_suite_replays_judges_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verify, "run_all", _fake_run_all(["complied", "refused", "complied", "error", "complied"]))
    provider = FakeTraceProvider([t for t in TRACES if t is not None])

    result = asyncio.run(
        verify.verify_suite(REQUESTS, None, TraceWindow(provider, None), known_tools=["read_balance"])  # type: ignore[arg-type]
    )

    assert result.traced
    assert (result.rows, result.traced_rows) == (5, 4)
    assert result.by_status("missing") == ["teleport"]
    assert result.by_status("unknown") == ["summon_cat"]
    assert list(result.observed_unlisted) == ["lookup_contact"]  # suite tools count as known
    assert result.baseline_refused == [1]
    assert result.row_results[0]["tool_facts"]["trace_status"] == "inspected"


def test_verify_suite_without_traces_leaves_every_tool_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    class Broken(FakeTraceProvider):
        def get_traces(self) -> list[AgentTrace]:
            raise RuntimeError("sandbox gone")

    monkeypatch.setattr(verify, "run_all", _fake_run_all(["complied"] * len(REQUESTS)))

    result = asyncio.run(
        verify.verify_suite(REQUESTS, None, TraceWindow(Broken(), None), known_tools=[])  # type: ignore[arg-type]
    )

    assert not result.traced
    assert result.traced_rows == 0
    assert {v.status for v in result.tools} == {"unknown"}
    assert result.observed_unlisted == {}
