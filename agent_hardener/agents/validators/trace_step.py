# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The piece both validators share for their trace step.

Each validator stays one module with one ``run``: it replays first, then continues into its trace
section whenever ``ctx.trace_provider`` is set (always, except for a plain HTTP target with no
relay-victim block). That section only adds ``tool_facts`` / ``trace_verdict`` to the results; it never
overwrites ``verdict``. Reports count :func:`final_status`, which falls back to the replay's own status,
so a result with no trace, or a trace step that failed, reports what the replay alone said.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agent_hardener.traces import trace_clock, traces_since

if TYPE_CHECKING:
    from datetime import datetime

    from agent_hardener.runtime.run_context import RunContext
    from agent_hardener.traces import AgentTrace, TraceProvider

logger = logging.getLogger(__name__)

#: ``tool_facts`` for an item no trace matched.
UNTRACED: dict[str, str] = {"trace_status": "untraced"}


@dataclass(frozen=True, slots=True)
class TraceWindow:
    """The traffic a validator is about to send, bounded on the trace clock.

    Opened *before* the replay, because the anchor must predate the validator's own requests: with an
    append-only event log, everything earlier (the attacks, earlier iterations) then falls outside.
    """

    provider: TraceProvider
    anchor: datetime | None

    async def traces(self) -> list[AgentTrace]:
        """The traces that started after the anchor."""
        return traces_since(await asyncio.to_thread(self.provider.get_traces), self.anchor)


async def open_trace_window(ctx: RunContext | None) -> TraceWindow | None:
    """A window for this run's replays, or ``None`` to run without traces.

    ``None`` when the validator was given no provider, and when reading the anchor fails: without an
    anchor older traffic cannot be told apart, so the whole run falls back rather than guess.
    """
    provider = ctx.trace_provider if ctx is not None else None
    if provider is None:
        return None
    try:
        anchor = trace_clock(await asyncio.to_thread(provider.get_traces))
    except Exception:
        logger.warning("could not read traces before the replay; validating without them", exc_info=True)
        return None
    return TraceWindow(provider, anchor)


def final_status(result: dict[str, Any]) -> str:
    """The item's status: the trace's relabel when there is one, else the replay's own verdict."""
    return (result.get("trace_verdict") or {}).get("status") or result["verdict"]["status"]
