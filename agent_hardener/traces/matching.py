# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick the traces a consumer cares about out of everything a provider returns.

Providers return all traces; the rules for which one belongs to an attack or a benign request live
here, once, and read only :class:`AgentTrace` fields so they work for every backend.

**Time windows.** While the sandbox lives its event file is append-only, so one ``get_traces()`` also
returns every earlier phase's traffic. To keep only the traffic it is about to send, a consumer reads
``anchor = trace_clock(provider.get_traces())`` *before* sending, then ``traces_since(..., anchor)``
after. The anchor comes from trace timestamps (the sandbox clock), so host/sandbox clock skew cannot
shift the window.

**Claim once.** Both finders take an optional ``claimed`` set of trace ids. They skip traces already
in it and add the ones they return, so two identical prompts sent in one batch get two distinct
traces instead of the same one twice.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from agent_hardener.traces.models import AgentTrace

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Compare prompts and answers modulo whitespace, which transports and loggers reflow."""
    return _WHITESPACE.sub(" ", text or "").strip()


def trace_clock(traces: Iterable[AgentTrace]) -> datetime | None:
    """The latest ``ended_at`` among ``traces``, or ``None`` if there are none. Read before sending."""
    return max((trace.ended_at for trace in traces), default=None)


def traces_since(traces: Iterable[AgentTrace], since: datetime | None) -> list[AgentTrace]:
    """Traces that started strictly after ``since`` (all of them when ``since`` is ``None``)."""
    if since is None:
        return list(traces)
    return [trace for trace in traces if trace.started_at > since]


def find_attack_trace(
    traces: Iterable[AgentTrace],
    input: str,  # noqa: A002 - mirrors AgentTrace.input
    response: str,
    at: datetime | None = None,
    claimed: set[str] | None = None,
) -> AgentTrace | None:
    """The one attempt behind an attack hit.

    1. The normalized input must match.
    2. Among those, prefer the attempts whose output starts with the hit's response (the hitlog may
       clip it). If none does, the transport reshaped the answer, so every input match stays.
    3. If several remain, take the one that started closest to ``at``, else the latest.
    """
    candidates = _unclaimed_with_input(traces, input, claimed)
    if not candidates:
        return None
    wanted = normalize(response)
    if wanted:
        by_output = [trace for trace in candidates if normalize(trace.output).startswith(wanted)]
        candidates = by_output or candidates
    if at is not None:
        chosen = min(candidates, key=lambda trace: abs((trace.started_at - at).total_seconds()))
    else:
        chosen = max(candidates, key=lambda trace: trace.started_at)
    _claim(claimed, [chosen])
    return chosen


def find_benign_traces(
    traces: Iterable[AgentTrace],
    input: str,  # noqa: A002 - mirrors AgentTrace.input
    since: datetime | None = None,
    claimed: set[str] | None = None,
) -> list[AgentTrace]:
    """Every run of a benign request, oldest first. ``since`` keeps only runs that started after it."""
    found = sorted(
        _unclaimed_with_input(traces_since(traces, since), input, claimed), key=lambda trace: trace.started_at
    )
    _claim(claimed, found)
    return found


def _unclaimed_with_input(
    traces: Iterable[AgentTrace],
    input: str,  # noqa: A002 - mirrors AgentTrace.input
    claimed: set[str] | None,
) -> list[AgentTrace]:
    wanted = normalize(input)
    return [
        trace
        for trace in traces
        if normalize(trace.input) == wanted and (claimed is None or trace.trace_id not in claimed)
    ]


def _claim(claimed: set[str] | None, traces: Iterable[AgentTrace]) -> None:
    if claimed is not None:
        claimed.update(trace.trace_id for trace in traces)
