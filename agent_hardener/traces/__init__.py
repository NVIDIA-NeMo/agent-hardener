# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The victim's execution traces, behind one backend-neutral interface.

Nothing outside this package may know which backend recorded a trace. Consumers use
:class:`AgentTrace`, a :class:`TraceProvider`, and the helpers re-exported here.
"""

from __future__ import annotations

# Load the models package first. Contracts embed AgentTrace, so importing this package before
# ``agent_hardener.models`` would otherwise re-enter traces.models while it is still half-initialized.
import agent_hardener.models
from agent_hardener.traces.factory import build_trace_provider, uses_traces
from agent_hardener.traces.matching import find_attack_trace, find_benign_traces, trace_clock, traces_since
from agent_hardener.traces.models import AgentTrace, LlmCall, ToolCall, ToolOutcome, TraceStep
from agent_hardener.traces.provider import TraceProvider
from agent_hardener.traces.render import render_trace
from agent_hardener.traces.signals import classify_tool_call

__all__ = [
    "AgentTrace",
    "LlmCall",
    "ToolCall",
    "ToolOutcome",
    "TraceProvider",
    "TraceStep",
    "build_trace_provider",
    "classify_tool_call",
    "find_attack_trace",
    "find_benign_traces",
    "render_trace",
    "trace_clock",
    "traces_since",
    "uses_traces",
]
