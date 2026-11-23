# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Post-synth tool verification: does each tool in the benign suite really exist in the victim?

The synth DAG infers tools from a description, the repo, API probes and the interview, so some are invented.
Verification never reads the victim's code: it replays the suite against the live victim through the benign
validator's own replay + judge + trace path, and reads which tools the traces show being called.

A tool is ``verified`` when *any* trace in the replay calls it (an agent may reach a tool from another tool's
request), ``missing`` when its rows were traced and no trace called it, and ``unknown`` when none of its rows
produced a trace. Tools the traces show but the profile does not list are collected with their calls, so the
caller can describe them and generate rows for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from agent_hardener.models.base import AgentHardenerModel
from agent_hardener.traces.inspect import same_tool
from agent_hardener.traces.signals import text_of

from .config import defaults
from .models import ToolStatus
from .replay_judge import run_all
from .replay_judge.traces import trace_rows

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Iterable

    from agent_hardener.agents.validators.trace_step import TraceWindow
    from agent_hardener.traces import AgentTrace

    from .models import GeneratedRequest
    from .replay_judge import ReplayJudgeConfig

_EXCERPT = 300


class ObservedCall(AgentHardenerModel):
    """One call of a tool the profile does not list, with the request that led to it."""

    tool: str
    args: Any = None
    result_excerpt: str = ""
    prompt: str


class ToolVerdict(AgentHardenerModel):
    """What the replay said about one suite tool, with the counts it rests on."""

    tool: str
    status: ToolStatus
    #: The tool's own rows, how many of them produced a trace, and how many of those called it.
    rows: int
    traced_rows: int
    rows_calling_it: int
    #: Traced rows of any tool whose trace called it: the evidence a ``verified`` rests on.
    calls_anywhere: int


class VerificationResult(AgentHardenerModel):
    """One verification pass over a set of suite rows."""

    #: False when the trace step failed: every tool is then ``unknown``.
    traced: bool
    rows: int
    traced_rows: int
    tools: list[ToolVerdict]
    observed_unlisted: dict[str, list[ObservedCall]] = Field(default_factory=dict)
    #: Indexes (into the verified rows) the unhardened victim refused, per the judge.
    baseline_refused: list[int] = Field(default_factory=list)
    #: The replay + judge + trace rows, as the in-loop validator would report them.
    row_results: list[dict[str, Any]] = Field(default_factory=list)

    def by_status(self, status: ToolStatus) -> list[str]:
        """The tools with ``status``, in suite order."""
        return [verdict.tool for verdict in self.tools if verdict.status == status]


async def verify_suite(
    requests: list[GeneratedRequest],
    cfg: ReplayJudgeConfig,
    window: TraceWindow,
    *,
    known_tools: Iterable[str],
    args_judge: Callable[[str, str, Any], Awaitable[bool]] | None = None,
) -> VerificationResult:
    """Replay ``requests`` against the victim, judge each, and verify their tools from the traces.

    ``window`` must be opened before this call so the traces are only this replay's. ``known_tools`` are the
    names already in the profile; any other tool the traces show lands in ``observed_unlisted``.
    """
    settings = defaults().verification
    row_results = await run_all(requests, cfg)
    matched = await trace_rows(row_results, requests, window, args_judge)
    row_traces: list[AgentTrace | None] = matched if matched is not None else [None] * len(requests)
    known = [*known_tools, *(r.tool for r in requests), *settings.ignore_tools]
    return VerificationResult(
        traced=matched is not None,
        rows=len(requests),
        traced_rows=sum(trace is not None for trace in row_traces),
        tools=classify_tools(requests, row_traces),
        observed_unlisted=unlisted_tools(known, requests, row_traces, settings.max_observed_calls_per_tool),
        baseline_refused=[i for i, result in enumerate(row_results) if result["verdict"]["status"] == "refused"],
        row_results=row_results,
    )


def classify_tools(requests: list[GeneratedRequest], row_traces: list[AgentTrace | None]) -> list[ToolVerdict]:
    """One verdict per suite tool, in order of first appearance."""
    traces = [trace for trace in row_traces if trace is not None]
    verdicts: list[ToolVerdict] = []
    for tool in dict.fromkeys(request.tool for request in requests):
        own = [trace for request, trace in zip(requests, row_traces, strict=True) if request.tool == tool]
        own_traced = [trace for trace in own if trace is not None]
        calls_anywhere = sum(_calls(trace, tool) for trace in traces)
        verdicts.append(
            ToolVerdict(
                tool=tool,
                status=_status(calls_anywhere, len(own_traced)),
                rows=len(own),
                traced_rows=len(own_traced),
                rows_calling_it=sum(_calls(trace, tool) for trace in own_traced),
                calls_anywhere=calls_anywhere,
            )
        )
    return verdicts


def unlisted_tools(
    known_tools: Iterable[str],
    requests: list[GeneratedRequest],
    row_traces: list[AgentTrace | None],
    max_calls: int,
) -> dict[str, list[ObservedCall]]:
    """Tools the traces show that match none of ``known_tools``, each with up to ``max_calls`` calls.

    Keyed by the bare tool name (namespace dropped), in order of first appearance.
    """
    known = list(known_tools)
    observed: dict[str, list[ObservedCall]] = {}
    for request, trace in zip(requests, row_traces, strict=True):
        if trace is None:
            continue
        for call in trace.tool_calls:
            if any(same_tool(call.name, name) for name in known):
                continue
            calls = observed.setdefault(call.name.rsplit(".", 1)[-1], [])
            if len(calls) < max_calls:
                calls.append(
                    ObservedCall(
                        tool=call.name,
                        args=call.args,
                        result_excerpt=text_of(call.result)[:_EXCERPT],
                        prompt=request.payload,
                    )
                )
    return observed


def _calls(trace: AgentTrace, tool: str) -> bool:
    return any(same_tool(call.name, tool) for call in trace.tool_calls)


def _status(calls_anywhere: int, own_traced: int) -> ToolStatus:
    if calls_anywhere:
        return "verified"
    return "missing" if own_traced else "unknown"
