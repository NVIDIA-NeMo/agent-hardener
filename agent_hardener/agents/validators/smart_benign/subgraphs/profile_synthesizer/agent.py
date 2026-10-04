# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the profile_synthesizer agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.merge import merge
from .nodes.normalize import normalize
from .state import ProfileSynthesizerState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_profile_synthesizer_agent() -> CompiledStateGraph:
    """Build and compile the profile_synthesizer agent.

    Two-node sequential graph: a deterministic ``merge`` followed by an
    LLM-backed ``normalize``. Growth (e.g., per-tool re-ranking, conflict
    detection that loops back to the interviewer) is purely additive.
    """
    graph: StateGraph = StateGraph(ProfileSynthesizerState)
    graph.add_node("merge", merge)
    graph.add_node("normalize", normalize)
    graph.add_edge(START, "merge")
    graph.add_edge("merge", "normalize")
    graph.add_edge("normalize", END)
    return graph.compile()
