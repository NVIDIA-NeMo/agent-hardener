# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the gap_detector agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.analyze_boundaries import analyze_boundaries
from .nodes.detect_gaps import detect_gaps
from .state import GapDetectorState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_gap_detector_agent() -> CompiledStateGraph:
    """Build and compile the gap_detector agent.

    Two nodes run in sequence: the deterministic ``detect_gaps`` rules, then the
    LLM ``analyze_boundaries`` pass that appends ``boundary[<tool>]:`` gaps about
    each tool's benign/attack line. The boundary pass no-ops (no LLM call) when
    the run is non-interactive, so non-TTY runs stay rules-only.
    """
    graph: StateGraph = StateGraph(GapDetectorState)
    graph.add_node("detect_gaps", detect_gaps)
    graph.add_node("analyze_boundaries", analyze_boundaries)
    graph.add_edge(START, "detect_gaps")
    graph.add_edge("detect_gaps", "analyze_boundaries")
    graph.add_edge("analyze_boundaries", END)
    return graph.compile()
