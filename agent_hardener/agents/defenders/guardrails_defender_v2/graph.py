# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

from langgraph.graph import END, START, StateGraph

from .nodes.component_writer import component_writer_node
from .nodes.custom_guardrails_generator import custom_guardrails_generator_node
from .nodes.data_augmenter import data_augmenter_node
from .nodes.vulnerability_analyzer import vulnerability_analyzer_node
from .schemas import GraphState


def build_defender_graph():
    """Constructs and compiles the LangGraph workflow for the Guardrails Defender."""
    # 1. Initialize Graph with State
    workflow = StateGraph(GraphState)

    # 2. Add Nodes
    workflow.add_node("init", _init_node)
    workflow.add_node("data_augmenter", data_augmenter_node)
    workflow.add_node("vulnerability_analyzer", vulnerability_analyzer_node)
    workflow.add_node("guardrail_generator", custom_guardrails_generator_node)
    # workflow.add_node("local_validator", local_validator_node)
    # workflow.add_node("feedback_analyzer", feedback_analyzer_node)
    workflow.add_node("component_writer", component_writer_node)

    # 3. Define Flow & Edges
    workflow.add_edge(START, "init")

    # --- Parallel Fork directly from START ---
    # workflow.add_edge(START, "data_augmenter")
    workflow.add_edge("init", "vulnerability_analyzer")

    # --- Merge Parallel Branches ---
    # workflow.add_edge("data_augmenter", "guardrail_generator")
    workflow.add_edge("vulnerability_analyzer", "guardrail_generator")

    # Currently not using local validator
    # @ TODO: Check if local validation is needed
    workflow.add_edge("guardrail_generator", "component_writer")
    workflow.add_edge("component_writer", END)

    return workflow.compile()


#     # --- Generation to Validation ---
#     workflow.add_edge("guardrail_generator", "local_validator")

#     # --- Conditional Turnaround Loop ---
#     workflow.add_conditional_edges(
#         "local_validator", route_evaluation, {"commit": "component_writer", "feedback": "feedback_analyzer"}
#     )

#     # --- Feedback: retry or hard-stop directly to component_writer ---
#     workflow.add_conditional_edges(
#         "feedback_analyzer",
#         route_feedback,
#         {
#             "retry": "guardrail_generator",
#             "stop": "component_writer",
#         },
#     )

#     # --- End ---
#     workflow.add_edge("component_writer", END)

#     # 4. Compile the Graph
#     return workflow.compile()


# def route_evaluation(state: GraphState) -> str:
#     if state.eval_passed:
#         return "commit"
#     return "feedback"


# def route_feedback(state: GraphState) -> str:
#     # if state.retry_counter > 3:
#     #     return "stop"
#     # return "retry"
#     return "stop"


def _init_node(state: GraphState) -> dict[str, Any]:
    """Copies relay_plugins_path from the typed input field into graph state."""
    path = state.original_input.relay_plugins_path
    return {"relay_plugins_path": path if (path and path.exists()) else None}
