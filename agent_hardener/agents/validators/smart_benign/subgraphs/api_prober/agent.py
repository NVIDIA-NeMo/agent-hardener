# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the api_prober agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.extract import extract
from .nodes.probe import probe
from .state import ApiProberState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_api_prober_agent() -> CompiledStateGraph:
    """Build and compile the api_prober agent.

    Two-node sequential graph: ``probe`` (bounded chat-completions HTTP
    requests to the target) followed by ``extract`` (one LLM call that
    turns the probe transcript into typed ``ToolSpec`` hypotheses tagged
    ``source="api_probe"``).
    """
    graph: StateGraph = StateGraph(ApiProberState)
    graph.add_node("probe", probe)
    graph.add_node("extract", extract)
    graph.add_edge(START, "probe")
    graph.add_edge("probe", "extract")
    graph.add_edge("extract", END)
    return graph.compile()
