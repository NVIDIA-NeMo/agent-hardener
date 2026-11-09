# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The victim's execution traces, behind one backend-neutral interface.

Nothing outside this package may know which backend recorded a trace. Consumers use
:class:`AgentTrace`, a :class:`TraceProvider`, and the helpers re-exported here. The trace models
themselves live in :mod:`agent_hardener.models.traces`, because the core contracts embed them.
"""

from __future__ import annotations

from agent_hardener.models.traces import AgentTrace, LlmCall, ToolCall, ToolOutcome, TraceStep
from agent_hardener.traces.factory import build_trace_provider
from agent_hardener.traces.matching import find_attack_trace, find_benign_traces, trace_clock, traces_since
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
]
