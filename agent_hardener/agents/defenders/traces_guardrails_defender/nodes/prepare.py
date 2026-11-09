# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn the attack trace into what the analyzer reads: rendered text and the tool calls to choose from."""

from __future__ import annotations

from typing import Any

from agent_hardener.models.traces import ToolCall
from agent_hardener.traces.render import render_trace

from ..schemas import CandidateCall
from ..state import DefenderState  # noqa: TC001 - LangGraph reads node annotations at runtime

#: Tool-call outcomes the analyzer may choose from: every call the attack drove the agent to make.
#: A call a guardrail already refused is left out: it is guarded.
ATTEMPTED_OUTCOMES = ("ok", "error", "network_denied")

#: Per-payload clip for the rendered trace. Wider than ``render_trace``'s default: the arguments are
#: the evidence the analyzer has to point at.
TRACE_CLIP = 600


def prepare_node(state: DefenderState) -> dict[str, Any]:
    """Render the trace and list the tool calls the analyzer may choose from.

    A call that failed or was denied egress is still a candidate: the attack drove the agent to make it,
    and a rail in front of that tool stops the attempt the next time it is made.
    """
    trace = state.input.attack_trace
    if trace is None:  # unreachable: the entry point fails an attack that has no trace
        msg = "prepare_node needs the attack's trace"
        raise ValueError(msg)
    candidates = [
        CandidateCall(step=index, tool_name=step.name, args=step.args, outcome=step.outcome)
        for index, step in enumerate(trace.steps, start=1)
        if isinstance(step, ToolCall) and step.outcome in ATTEMPTED_OUTCOMES
    ]
    return {"rendered_trace": render_trace(trace, limit=TRACE_CLIP), "candidates": candidates}
