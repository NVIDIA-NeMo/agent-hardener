# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the nl_parser agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.parse import parse
from .state import NLParserState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_nl_parser_agent() -> CompiledStateGraph:
    """Build and compile the nl_parser agent.

    The agent is implemented as a LangGraph: today it wraps a single LangChain
    ``with_structured_output`` call inside the ``parse`` node. Growth (e.g.,
    decompose long descriptions, ask follow-up sub-questions, run multiple
    parsers in parallel) is purely additive — add nodes/edges inside this
    factory; the parent graph never changes.
    """
    graph: StateGraph = StateGraph(NLParserState)
    graph.add_node("parse", parse)
    graph.add_edge(START, "parse")
    graph.add_edge("parse", END)
    return graph.compile()
