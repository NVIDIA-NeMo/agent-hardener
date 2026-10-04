# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the request_generator agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.generate import generate
from .state import RequestGeneratorState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_request_generator_agent() -> CompiledStateGraph:
    """Build and compile the request_generator agent.

    One LangGraph node: per-tool LLM calls run in parallel under
    ``gather_limited_ordered`` with bounded concurrency. Pydantic schema
    validation enforces per-row types; the ``critic`` agent downstream
    handles dedup and off-topic filtering.
    """
    graph: StateGraph = StateGraph(RequestGeneratorState)
    graph.add_node("generate", generate)
    graph.add_edge(START, "generate")
    graph.add_edge("generate", END)
    return graph.compile()
