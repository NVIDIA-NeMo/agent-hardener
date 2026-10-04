# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the critic agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.critique import critique
from .state import CriticState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_critic_agent() -> CompiledStateGraph:
    """Build and compile the critic agent.

    One LangGraph node: an LLM critique pass followed by a deterministic
    exact-match dedup. Growth (e.g., per-tool critic specializations, a second
    semantic-dedup pass) is purely additive — add nodes inside this factory.
    """
    graph: StateGraph = StateGraph(CriticState)
    graph.add_node("critique", critique)
    graph.add_edge(START, "critique")
    graph.add_edge("critique", END)
    return graph.compile()
