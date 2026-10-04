# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent synth graph: wires the 9 sub-agents into the smart benign validator's DAG.

Topology:

    START → [nl_parser, github_analyzer, api_prober]   (parallel ingestion)
            └─→ profile_synthesizer
                └─→ gap_detector
                    ├─ gaps && interactive && questions_asked < budget → interview_compose → interview_ask → profile_synthesizer (loop)
                    └─ otherwise                                       → request_generator → critic → profile_writer → END

The interview is split across two parent nodes: ``interview_compose`` (LLM builds the question batch) and
``interview_ask`` (``interrupt()`` for answers, when the graph is compiled with a checkpointer; the driver
supplies them). The loop is bounded by two budgets: ``SynthState.max_interview_questions`` (default 10) and
``SynthState.max_interview_rounds`` (default 2); ``interview_ask`` advances both counters each round so the
loop terminates. Each round re-runs ``profile_synthesizer`` + ``gap_detector``, so the round cap directly
bounds repeated re-synthesis; on the closing pass the boundary fan-out no-ops (``interview_open`` is False).
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .state import SynthState
from .subgraphs import (
    api_prober,
    critic,
    gap_detector,
    github_analyzer,
    interviewer,
    nl_parser,
    profile_synthesizer,
    profile_writer,
    request_generator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from langchain_core.runnables import RunnableConfig
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph


@contextmanager
def quiet_warnings() -> Iterator[None]:
    """Suppress all warnings for the wrapped block.

    Synth and replay emit several harmless-but-noisy third-party warnings (Pydantic v2
    serializer warnings on LangChain's structured-output wrapper, ChatNVIDIA's "model not
    known to support structured output", etc.). Re-applying the blanket filter inside this
    context keeps it active even if a dependency reset ``warnings.filters`` earlier in the
    process — which a plain module-level filter does not survive.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


def _wrap(agent: Any, state_type: Any, entry: Callable, exit_: Callable) -> Callable:
    """Adapt a compiled subgraph into a parent-graph node."""

    async def node(state: SynthState, config: RunnableConfig) -> dict[str, object]:
        return exit_(state_type.model_validate(await agent.ainvoke(entry(state), config)))

    return node


def _ingestion_sources(state: SynthState) -> list[str]:
    sources = []
    if not state.inputs.skip_nl_parser:
        sources.append("nl_parser")
    if not state.inputs.skip_github_analysis:
        sources.append("github_analyzer")
    if not state.inputs.skip_api_probe:
        sources.append("api_prober")
    # No active sources → skip straight to profile_synthesizer so the
    # gap detector fires and the interview loop fills the profile.
    return sources or ["profile_synthesizer"]


def _route_after_gap_detector(state: SynthState) -> str:
    answered_gaps = {gap for gap, _q, _a in state.interviewer_answers if gap}
    unresolved = [g for g in state.gaps if g not in answered_gaps]
    can_ask = (
        state.interactive
        and state.interview_rounds < state.max_interview_rounds
        and state.interview_questions_asked < state.max_interview_questions
    )
    if unresolved and can_ask:
        return "interview_compose"
    return "request_generator"


def _interview_compose_node(agent: Any) -> Callable:
    """Run the compose subgraph to stage this round's questions on ``composed_questions``."""

    async def node(state: SynthState, config: RunnableConfig) -> dict[str, object]:
        sub_out = await agent.ainvoke(interviewer.entry_adapter(state), config)
        return interviewer.exit_adapter(interviewer.InterviewerState.model_validate(sub_out))

    return node


def _interview_ask_node(state: SynthState) -> dict[str, object]:
    """Interrupt for answers to the staged questions, record them, advance the budget, clear the stage.

    Kept separate from compose so a resume re-runs only this node. An empty batch (all gaps answered) bumps
    the budget to the ceiling to exit the loop without interrupting.
    """
    questions = state.composed_questions
    if not questions:
        return {"interview_questions_asked": state.max_interview_questions, "composed_questions": []}
    answers = interrupt({"questions": questions})
    by_gap = {q["gap"]: q.get("question", "") for q in questions}
    collected = [
        (str(a.get("gap", "")), a.get("question") or by_gap.get(a.get("gap", ""), ""), str(a["answer"]).strip())
        for a in (answers or [])
        if str(a.get("answer", "")).strip()
    ]
    asked = state.interview_questions_asked + max(len(collected), len(questions), 1)
    return {
        "interviewer_answers": collected,
        "interview_questions_asked": asked,
        "interview_rounds": state.interview_rounds + 1,
        "composed_questions": [],
    }


def build_synth_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:
    """Build and compile the parent synth graph.

    Pass a ``checkpointer`` (e.g. ``MemorySaver``) to enable the interview ``interrupt()``/resume; without
    one the graph runs straight through (the router still only enters the interview when ``interactive``).
    """
    nodes = {
        "nl_parser": _wrap(
            nl_parser.build_nl_parser_agent(),
            nl_parser.NLParserState,
            nl_parser.entry_adapter,
            nl_parser.exit_adapter,
        ),
        "github_analyzer": _wrap(
            github_analyzer.build_github_analyzer_agent(),
            github_analyzer.GitHubAnalyzerState,
            github_analyzer.entry_adapter,
            github_analyzer.exit_adapter,
        ),
        "api_prober": _wrap(
            api_prober.build_api_prober_agent(),
            api_prober.ApiProberState,
            api_prober.entry_adapter,
            api_prober.exit_adapter,
        ),
        "profile_synthesizer": _wrap(
            profile_synthesizer.build_profile_synthesizer_agent(),
            profile_synthesizer.ProfileSynthesizerState,
            profile_synthesizer.entry_adapter,
            profile_synthesizer.exit_adapter,
        ),
        "gap_detector": _wrap(
            gap_detector.build_gap_detector_agent(),
            gap_detector.GapDetectorState,
            gap_detector.entry_adapter,
            gap_detector.exit_adapter,
        ),
        "interview_compose": _interview_compose_node(interviewer.build_interviewer_agent()),
        "interview_ask": _interview_ask_node,
        "request_generator": _wrap(
            request_generator.build_request_generator_agent(),
            request_generator.RequestGeneratorState,
            request_generator.entry_adapter,
            request_generator.exit_adapter,
        ),
        "critic": _wrap(
            critic.build_critic_agent(),
            critic.CriticState,
            critic.entry_adapter,
            critic.exit_adapter,
        ),
        "profile_writer": _wrap(
            profile_writer.build_profile_writer_agent(),
            profile_writer.ProfileWriterState,
            profile_writer.entry_adapter,
            profile_writer.exit_adapter,
        ),
    }

    graph: StateGraph = StateGraph(SynthState)
    for name, fn in nodes.items():
        graph.add_node(name, fn)

    # Parallel ingestion fan-in → profile_synthesizer
    # Only dispatch sources whose skip flag is not set.
    graph.add_conditional_edges(
        START, _ingestion_sources, ["nl_parser", "github_analyzer", "api_prober", "profile_synthesizer"]
    )
    for source in ("nl_parser", "github_analyzer", "api_prober"):
        graph.add_edge(source, "profile_synthesizer")

    graph.add_edge("profile_synthesizer", "gap_detector")
    graph.add_conditional_edges(
        "gap_detector",
        _route_after_gap_detector,
        {"interview_compose": "interview_compose", "request_generator": "request_generator"},
    )
    graph.add_edge("interview_compose", "interview_ask")
    graph.add_edge("interview_ask", "profile_synthesizer")  # re-merge with new answers

    graph.add_edge("request_generator", "critic")
    graph.add_edge("critic", "profile_writer")
    graph.add_edge("profile_writer", END)

    return graph.compile(checkpointer=checkpointer)
