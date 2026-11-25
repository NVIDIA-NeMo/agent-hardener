# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the victim is NeMo Relay-connected before the first attack is sent.

Agent Hardener reads the victim's tool calls from Relay's ATOF stream (:mod:`agent_hardener.atof`)
and delivers guardrails as Relay plugin config. Both require the agent to attach
``NemoRelayMiddleware`` *and* call ``nemo_relay.plugin.initialize(...)`` at startup — nothing
activates a delivered ``/etc/nemo-relay/plugins.toml`` without that call.

This check is load-bearing rather than a nicety. An uninstrumented victim answers every attack and
emits no ATOF, so the run looks like a working demo of a guardrail that was never installed. Failing
here costs one probe; failing to fail here costs the credibility of the whole report.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_hardener.atof import has_quarantined_scopes, load_records, model_scope_ends, requested_tools, tool_calls
from agent_hardener.errors import VictimNotInstrumentedError, VictimUnavailableError

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Collection, Iterable
    from pathlib import Path

#: A greeting, not a tool request: this proves Relay is attached. Whether tool calls *reach* it is a
#: different question, answered later from real attack traffic (:func:`check_tool_path`).
RELAY_PROBE_PROMPT = "Hello — briefly, what can you help with?"
RELAY_PROBE_TIMEOUT_SECONDS = 60.0

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_POLL_INTERVAL_SECONDS = 0.5


@dataclass(frozen=True, slots=True)
class RelayPreflight:
    """What one warm-up probe proved about the victim's instrumentation."""

    records_observed: int
    tools_seen: tuple[str, ...]
    quarantined: bool


def check_tool_path(records: Iterable[dict[str, Any]], *, before: Collection[str] = ()) -> tuple[str, ...]:
    """Return observed tools; an empty tuple means tool mediation remains unverified.

    Separate from the probe check because the evidence arrives much later. A warm-up greeting proves
    Relay is *attached*; only real attack traffic proves tool calls *pass through* it. A victim can
    pass the first and fail this one — a hand-built LangGraph graph emits LLM scopes happily while
    its tool calls bypass Relay entirely — and that victim is unguardable no matter what the
    defenders write.

    "No tool call" has three causes, told apart by the victim's own model calls since the startup probe:

    - the model asked for tools and none ran through Relay: the tools bypass it (not instrumented);
    - no model call was traced at all: Relay stopped tracing after the probe (not instrumented);
    - model calls were traced but none asked for a tool: Relay is fine, the attacks never led the
      model to a tool (return an empty tuple, permitting continuation without verifying mediation).

    An attacker that sent nothing, or reached a dead victim, has already failed the attack stage.

    Missing telemetry and requested-but-unobserved tools still raise. A refusal without tool
    requests is valid attack traffic, but cannot establish that tool guardrails were exercised.

    Args:
        records: The victim's whole ATOF stream.
        before: The scope ids already in the stream when the startup probe passed, so the probe's own
            model call is not counted as attack traffic. By id, not position: a Fabric victim's stream is
            several per-session files joined in directory order, not in time order.
    """
    records = list(records)
    traffic = [record for record in records if record.get("uuid") not in before]
    seen = tuple(sorted({call.name for call in tool_calls(traffic) if call.name}))
    if seen:
        return seen
    if requested := requested_tools(traffic):
        asked = ", ".join(f"{name} x{count}" for name, count in Counter(requested).most_common())
        raise VictimNotInstrumentedError(
            f"the victim's model asked for tools ({asked}) but Relay recorded none of them running, so its "
            "tool calls do not pass through Relay — a guardrail installed here could never fire. Attach "
            "Relay's middleware (or, for a hand-built LangGraph graph, build the tools node with Relay's "
            "create_tool_node).",
            # Relay is attached and writing events here, so the class's "attach Relay" default misleads.
            remediation=(
                "Route the tool calls through Relay: for a hand-built LangGraph graph, "
                "ToolNode(TOOLS, awrap_tool_call=NemoRelayMiddleware().awrap_tool_call) or Relay's create_tool_node."
            ),
        )
    model_calls = len(model_scope_ends(traffic))
    if model_calls == 0:
        raise VictimNotInstrumentedError(
            f"the victim answered attacks but Relay recorded no event for them ({len(traffic)} record(s) since "
            "the startup probe, no model call among them), so tracing stopped after the probe or the "
            "attacks were served by a process Relay is not attached to.",
            remediation=(
                "Check the victim log for a Relay error after startup, and that every worker process serving "
                "requests calls nemo_relay.plugin.initialize() and runs inside a Relay scope."
            ),
        )
    return ()


async def check_relay_instrumentation(
    *,
    probe: Callable[[], Awaitable[Any]],
    atof_path: Path,
    sync: Callable[[], Any] | None = None,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
) -> RelayPreflight:
    """Send one benign probe and require the victim to emit ATOF for it.

    Args:
        probe: Sends a single benign request to the victim. Only its HTTP status is read (a
            rejected probe is reported as a dead victim); the telemetry it provokes is the point.
        atof_path: Where the victim's ``events.atof.jsonl`` is readable from *here*.
        sync: Called before each read. A sandboxed victim shares no filesystem with the host, so its
            telemetry has to be copied over before it can be seen; without this the check would time
            out on every sandboxed run and report a correctly-instrumented victim as broken.
        timeout_seconds: How long to wait for records to land after the probe returns. Relay's
            subscribers flush asynchronously, so the file lags the response.
        poll_interval_seconds: Gap between reads while waiting.

    Returns:
        What the probe observed.

    Raises:
        VictimUnavailableError: The victim rejected the probe, so its agent never started.
        VictimNotInstrumentedError: No new ATOF records appeared, so the victim is not Relay-connected.
    """
    if sync is not None:
        # With a sync the local file is a *copy*, and a previous run leaves one behind. Drop it before
        # the first read: a fresh victim starts a new stream — Fabric writes it to a new
        # runtime-scoped path entirely — so last run's total is the wrong thing to measure against,
        # and a shorter new stream would never exceed it. Without a sync the file is the victim's own
        # output, where earlier records legitimately set the floor for a mid-run re-check.
        atof_path.unlink(missing_ok=True)
        sync()
    baseline = _record_count(atof_path)
    _require_probe_answered(await probe())

    deadline = time.monotonic() + timeout_seconds
    while True:
        if sync is not None:
            sync()
        records = load_records(atof_path) if atof_path.exists() else []
        if len(records) > baseline:
            return RelayPreflight(
                records_observed=len(records) - baseline,
                tools_seen=tuple(sorted({call.name for call in tool_calls(records) if call.name})),
                quarantined=has_quarantined_scopes(records),
            )
        if time.monotonic() >= deadline:
            raise VictimNotInstrumentedError(_diagnose(atof_path, baseline, timeout_seconds))
        await asyncio.sleep(poll_interval_seconds)


def _record_count(atof_path: Path) -> int:
    """Records already present, so a re-check mid-run does not count an earlier round's events."""
    return len(load_records(atof_path)) if atof_path.exists() else 0


def _require_probe_answered(response: Any) -> None:
    """Fail on a probe the victim rejected, before blaming Relay for the silence it causes.

    A victim whose agent never starts still answers — with a 5xx — and emits no telemetry, which is
    indistinguishable from a working victim that is simply not Relay-connected. Reporting the status
    and body here points at the agent instead of sending the reader after an instrumentation problem
    they do not have.
    """
    status = getattr(response, "status_code", None)
    if not isinstance(status, int) or 200 <= status < 300:
        return
    body = getattr(response, "text", "") or ""
    raise VictimUnavailableError(f"the victim answered the Relay probe with HTTP {status}: {body[:500]}")


def _diagnose(atof_path: Path, baseline: int, timeout_seconds: float) -> str:
    """Separate 'never wrote anything' from 'wrote before, went quiet' — different root causes."""
    if not atof_path.exists():
        return (
            f"the victim answered a probe but wrote no Relay telemetry to {atof_path} "
            f"within {timeout_seconds:g}s (the file was never created)"
        )
    if baseline == 0:
        return (
            f"the victim answered a probe but {atof_path} is still empty after {timeout_seconds:g}s; "
            "Relay is not attached, or its ATOF sink is disabled (AtofConfig.enabled defaults to false)"
        )
    return (
        f"the victim answered a probe but emitted no new Relay events in {timeout_seconds:g}s "
        f"({baseline} older records present), so this invocation went untraced"
    )
