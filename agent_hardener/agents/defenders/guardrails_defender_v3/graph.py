# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The v3 defender graph: vulnerability analysis, then rail generation on the route it chose.

Each step is a node in ``nodes/``:

- ``prepare`` renders the trace and lists the completed tool calls.
- ``vulnerability_analyzer`` grounds the harmful call in the trace (an answer that does not match it is asked again
  once), or falls back to the user prompt when no tool call carried the harm.
- tool route: ``benign_args`` → ``guardrail_generator``, which runs the suggestor ⇄ grader loop.
- no tool call carried the harm: ``skip``. A tool rail cannot separate such an attack from ordinary
  use, so the defender abstains rather than write a rail that would refuse the tool's normal work.
- ``component_writer`` appends the best-graded draft to the run's plugin config.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from .nodes.benign_args import benign_args_node
from .nodes.component_writer import component_writer_node
from .nodes.guardrail_generator import guardrail_generator_node
from .nodes.prepare import prepare_node
from .nodes.skip import skip_node
from .nodes.vulnerability_analyzer import route_after_analysis, vulnerability_analyzer_node
from .observability import Recorder
from .state import DefenderState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_graph(llm: BaseChatModel, grader: BaseChatModel | None = None, recorder: Recorder | None = None) -> Any:
    """Compile the graph.

    ``llm`` serves the analyzer, the benign-args writer and the suggestor; ``grader`` grades the
    suggestor's drafts (``llm`` when not given); ``recorder`` captures each node's output.
    """
    recorder = recorder or Recorder(None)
    grader = grader or llm
    graph = StateGraph(DefenderState)
    graph.add_node("prepare", recorder.node("prepare", prepare_node))
    graph.add_node("vulnerability_analyzer", recorder.node("analyze", partial(vulnerability_analyzer_node, llm=llm)))
    graph.add_node("benign_args", recorder.node("benign_args", partial(benign_args_node, llm=llm)))
    graph.add_node(
        "guardrail_generator", recorder.node("generate", partial(guardrail_generator_node, llm=llm, grader=grader))
    )
    graph.add_node("component_writer", component_writer_node)
    graph.add_node("skip", skip_node)

    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "vulnerability_analyzer")
    graph.add_conditional_edges(
        "vulnerability_analyzer", route_after_analysis, ["vulnerability_analyzer", "benign_args", "skip"]
    )
    graph.add_edge("benign_args", "guardrail_generator")
    graph.add_edge("guardrail_generator", "component_writer")
    graph.add_edge("component_writer", END)
    graph.add_edge("skip", END)
    return graph.compile()
