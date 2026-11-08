# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a trace as compact text for an LLM prompt."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.traces.models import ToolCall
from agent_hardener.traces.signals import text_of

if TYPE_CHECKING:
    from agent_hardener.traces.models import AgentTrace

_DEFAULT_LIMIT = 400


def render_trace(trace: AgentTrace, limit: int = _DEFAULT_LIMIT) -> str:
    """One line per step, each payload clipped to ``limit`` characters.

    Model calls show only their answer: their input is the conversation so far, which the user
    prompt and the earlier steps already show.
    """
    lines = [f"USER: {_clip(trace.input, limit)}"]
    for index, step in enumerate(trace.steps, start=1):
        if isinstance(step, ToolCall):
            status = step.outcome if step.rail is None else f"{step.outcome} by {step.rail}"
            lines.append(
                f"{index}. TOOL {step.name}({_clip(text_of(step.args), limit)}) -> [{status}] "
                f"{_clip(text_of(step.result), limit)}"
            )
        else:
            lines.append(f"{index}. LLM -> {_clip(step.output, limit)}")
    if trace.guardrail_latched:
        lines.append("(a guardrail latched this request: later model calls got the canned blocked answer)")
    lines.append(f"ANSWER: {_clip(trace.output, limit)}")
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
