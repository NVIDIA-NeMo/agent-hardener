# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Factory for the interviewer agent (compiled LangGraph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langgraph.graph import END, START, StateGraph

from .nodes.compose import compose
from .state import InterviewerState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


def build_interviewer_agent() -> CompiledStateGraph:
    """Build and compile the interviewer agent.

    Compose-only (gaps → questions). The human step (interrupt for answers) lives in the parent graph so
    it can pause/resume via the checkpointer; answers are collected by the driver, not this subgraph.
    """
    graph: StateGraph = StateGraph(InterviewerState)
    graph.add_node("compose", compose)
    graph.add_edge(START, "compose")
    graph.add_edge("compose", END)
    return graph.compile()
