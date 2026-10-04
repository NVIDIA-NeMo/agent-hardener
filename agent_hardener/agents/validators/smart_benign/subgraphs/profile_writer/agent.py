# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the profile_writer agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.write import write
from .state import ProfileWriterState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_profile_writer_agent() -> CompiledStateGraph:
    """Build and compile the profile_writer agent.

    Single deterministic node: writes ``profile.json``, ``requests.csv``,
    per-source notes under ``sources/``, and ``input_hash.txt`` to the target
    directory. The exit_adapter surfaces only the writer's own note +
    errors; the parent graph carries the rest of state untouched.
    """
    graph: StateGraph = StateGraph(ProfileWriterState)
    graph.add_node("write", write)
    graph.add_edge(START, "write")
    graph.add_edge("write", END)
    return graph.compile()
