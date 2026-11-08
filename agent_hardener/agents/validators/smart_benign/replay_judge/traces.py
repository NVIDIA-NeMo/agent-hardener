# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The benign validator's trace section: runs after replay + judge, only when traces are on.

For every row with a matching replay trace it records ``tool_facts`` (did the row's tool run, and what
stopped any call that did not), whatever the judge said. It explains only the rows the judge saw
refused: ``trace_verdict`` names the blocked tool and whether our guardrail, the network policy, a
plain tool error or the agent itself refused. ``verdict`` itself is never touched.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.trace_step import UNTRACED
from agent_hardener.traces import find_benign_traces
from agent_hardener.traces.inspect import inspect_benign_block, tool_facts

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
    from agent_hardener.agents.validators.trace_step import TraceWindow
    from agent_hardener.traces import AgentTrace

logger = logging.getLogger(__name__)

#: Trace statuses that blame a refusal on our own defenses: false positives fed back to defenders.
BLOCKED_BY_US = ("blocked_guardrail", "blocked_network")


async def apply_traces(results: list[dict[str, Any]], requests: list[GeneratedRequest], window: TraceWindow) -> bool:
    """Add ``tool_facts`` / ``trace_verdict`` to ``results`` (in request order), all or nothing.

    Returns whether it did. Any failure is logged and leaves ``results`` exactly as replay + judge
    produced them, so the report is the no-traces report.
    """
    try:
        updates = _updates(results, requests, await window.traces())
    except Exception:
        logger.warning("benign validator trace step failed; keeping the judge verdicts", exc_info=True)
        return False
    for result, update in zip(results, updates, strict=True):
        result.update(update)
    return True


def _updates(
    results: list[dict[str, Any]], requests: list[GeneratedRequest], traces: list[AgentTrace]
) -> list[dict[str, Any]]:
    claimed: set[str] = set()
    updates: list[dict[str, Any]] = []
    for result, request in zip(results, requests, strict=True):
        found = find_benign_traces(traces, request.payload, claimed=claimed)
        if not found:
            updates.append({"tool_facts": dict(UNTRACED)})
            continue
        # One replay per row; if the same payload appears twice, each row claims its own run.
        trace = found[0]
        claimed.difference_update(t.trace_id for t in found[1:])
        facts = tool_facts(trace, request.tool or None)
        update: dict[str, Any] = {"tool_facts": facts.model_dump(mode="json")}
        if result["verdict"]["status"] == "refused":
            update["trace_verdict"] = inspect_benign_block(facts).model_dump(mode="json", exclude_none=True)
        updates.append(update)
    return updates


def false_positive(result: dict[str, Any]) -> str | None:
    """The feedback line for a refusal caused by us (or an untraced refusal), else ``None``."""
    trace_verdict = result.get("trace_verdict") or {}
    excerpt = result.get("payload_excerpt", "")
    status = trace_verdict.get("status")
    if status == "blocked_guardrail":
        return f"{excerpt} — blocked by rail {trace_verdict.get('rail') or '(latched)'} on tool {_tool(trace_verdict)}"
    if status == "blocked_network":
        return f"{excerpt} — network denied on tool {_tool(trace_verdict)}"
    if status is None and result["verdict"]["status"] == "refused":
        return f"{excerpt} — refused (no trace to say by what)"
    return None


def _tool(trace_verdict: dict[str, Any]) -> str:
    return str(trace_verdict.get("blocked_tool") or "?")
