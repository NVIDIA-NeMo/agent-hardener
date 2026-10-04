# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the github_analyzer agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.clone_and_scan import clone_and_scan
from .nodes.summarize import summarize
from .state import GitHubAnalyzerState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_github_analyzer_agent() -> CompiledStateGraph:
    """Build and compile the github_analyzer agent.

    Two-node sequential graph:

    - ``clone_and_scan``: shallow ``git clone --depth 1`` into a
      ``TemporaryDirectory`` (whose lifetime is bounded by this node, so
      failures never leak), scan config files for declared tools,
      capture HEAD SHA + README text, then tear the dir down.
    - ``summarize``: one LLM call over the README to extract globals
      (``system_role`` / ``personas`` / ``out_of_scope``). The tool list
      stays authoritative from the YAML scan; the LLM cannot override it.
    """
    graph: StateGraph = StateGraph(GitHubAnalyzerState)
    graph.add_node("clone_and_scan", clone_and_scan)
    graph.add_node("summarize", summarize)
    graph.add_edge(START, "clone_and_scan")
    graph.add_edge("clone_and_scan", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()
