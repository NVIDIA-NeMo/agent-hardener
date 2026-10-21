# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent synth graph: node adapters, source fan-out, and the compiled DAG run end-to-end with scripted subgraphs."""

from __future__ import annotations

import asyncio
import warnings
from typing import TYPE_CHECKING, Any

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_hardener.agents.validators.smart_benign import graph
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.subgraphs import interviewer

if TYPE_CHECKING:
    from collections.abc import Callable


def _state(**kwargs: Any) -> SynthState:
    base: dict[str, Any] = {"inputs": SynthInputs(target_name="t")}
    base.update(kwargs)
    return SynthState(**base)


# --- quiet_warnings ----------------------------------------------------------------------------------------


def test_quiet_warnings_silences_warnings_only_inside_the_block() -> None:
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with graph.quiet_warnings():
            warnings.warn("noisy", UserWarning, stacklevel=1)
        warnings.warn("audible", UserWarning, stacklevel=1)

    assert [str(w.message) for w in seen] == ["audible"]


def test_quiet_warnings_restores_filters_when_the_block_raises() -> None:
    before = list(warnings.filters)

    with pytest.raises(RuntimeError), graph.quiet_warnings():
        raise RuntimeError("boom")

    assert warnings.filters == before


# --- _ingestion_sources ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("skips", "expected"),
    [
        ({}, ["nl_parser", "github_analyzer", "api_prober"]),
        ({"skip_nl_parser": True}, ["github_analyzer", "api_prober"]),
        ({"skip_github_analysis": True, "skip_api_probe": True}, ["nl_parser"]),
        (
            {"skip_nl_parser": True, "skip_github_analysis": True, "skip_api_probe": True},
            [
                "profile_synthesizer"
            ],  # nothing to ingest: go straight to synthesis so the interview can fill the profile
        ),
    ],
    ids=["all", "no-nl", "nl-only", "none"],
)
def test_ingestion_sources_follow_the_skip_flags(skips: dict[str, Any], expected: list[str]) -> None:
    state = _state(inputs=SynthInputs(target_name="t", **skips))

    assert graph._ingestion_sources(state) == expected


# --- _wrap / compose node ----------------------------------------------------------------------------------


class _FakeAgent:
    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.calls: list[tuple[Any, Any]] = []

    async def ainvoke(self, payload: Any, config: Any) -> dict[str, Any]:
        self.calls.append((payload, config))
        return self.output


class _Sub:
    """Minimal stand-in for a subgraph's private pydantic state."""

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def model_validate(cls, raw: dict[str, Any]) -> _Sub:
        return cls(raw["value"])


def test_wrap_runs_the_agent_between_the_entry_and_exit_adapters() -> None:
    agent = _FakeAgent({"value": 7})
    state = _state()
    config: Any = {"configurable": {"thread_id": "x"}}

    def entry(parent: SynthState) -> str:
        return f"entry:{parent.inputs.target_name}"

    def exit_(sub: _Sub) -> dict[str, object]:
        return {"gaps": [f"v{sub.value}"]}

    node = graph._wrap(agent, _Sub, entry, exit_)

    assert asyncio.run(node(state, config)) == {"gaps": ["v7"]}
    assert agent.calls == [("entry:t", config)]  # the run config (callbacks, thread id) reaches the subgraph


def test_wrap_propagates_subgraph_failures() -> None:
    class _Boom:
        async def ainvoke(self, *_a: Any) -> dict[str, Any]:
            raise RuntimeError("subgraph failed")

    node = graph._wrap(_Boom(), _Sub, lambda _s: None, lambda _s: {})

    with pytest.raises(RuntimeError, match="subgraph failed"):
        asyncio.run(node(_state(), {}))


def test_interview_compose_node_stages_serialised_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    question = {"gap": "g", "question": "Q?", "options": []}
    agent = _FakeAgent({"composed_questions": [question], "source_note": "n"})
    seen: list[SynthState] = []

    def entry(parent: SynthState) -> interviewer.InterviewerState:
        seen.append(parent)
        return interviewer.InterviewerState()

    monkeypatch.setattr(interviewer, "entry_adapter", entry)

    delta = asyncio.run(graph._interview_compose_node(agent)(_state(), {}))

    assert delta["composed_questions"] == [question]
    assert delta["source_notes"] == {"interviewer": "n"}
    assert len(seen) == 1


# --- _interview_ask_node -----------------------------------------------------------------------------------


def _ask(monkeypatch: pytest.MonkeyPatch, state: SynthState, answers: Any) -> dict[str, object]:
    monkeypatch.setattr(graph, "interrupt", lambda _payload: answers)
    return graph._interview_ask_node(state)


def test_ask_node_passes_the_staged_questions_to_the_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads: list[Any] = []
    questions = [{"gap": "g", "question": "Q?"}]

    def interrupt(payload: Any) -> list[Any]:
        payloads.append(payload)
        return []

    monkeypatch.setattr(graph, "interrupt", interrupt)

    graph._interview_ask_node(_state(composed_questions=questions))

    assert payloads == [{"questions": questions}]


def test_ask_node_drops_blank_answers_and_falls_back_to_the_composed_question_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(composed_questions=[{"gap": "g1", "question": "Q1?"}, {"gap": "g2", "question": "Q2?"}])
    answers = [
        {"gap": "g1", "answer": "  nurses  "},  # no question echoed back: use the composed text
        {"gap": "g2", "question": "Custom?", "answer": "   "},  # blank: dropped
        {"gap": "unknown", "answer": "x"},  # a gap that was never staged: empty question text
    ]

    out = _ask(monkeypatch, state, answers)

    assert out["interviewer_answers"] == [("g1", "Q1?", "nurses"), ("unknown", "", "x")]


@pytest.mark.parametrize("answers", [None, []], ids=["none", "empty"])
def test_ask_node_with_no_answers_still_spends_budget_so_the_loop_terminates(
    monkeypatch: pytest.MonkeyPatch, answers: Any
) -> None:
    state = _state(
        composed_questions=[{"gap": "g1", "question": "Q?"}, {"gap": "g2", "question": "R?"}],
        interview_questions_asked=3,
        interview_rounds=1,
    )

    out = _ask(monkeypatch, state, answers)

    assert out["interviewer_answers"] == []
    assert out["interview_questions_asked"] == 5  # charged for every question that was put to the user
    assert out["interview_rounds"] == 2
    assert out["composed_questions"] == []


def test_ask_node_clears_the_staged_questions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _state(composed_questions=[{"gap": "g", "question": "Q?"}])

    assert _ask(monkeypatch, state, [{"gap": "g", "answer": "A"}])["composed_questions"] == []


# --- the compiled DAG --------------------------------------------------------------------------------------


class _Script:
    """Replaces every subgraph node with a scripted function and records the execution order."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, gaps_per_pass: list[list[str]]) -> None:
        self.visited: list[str] = []
        self._gaps = list(gaps_per_pass)
        monkeypatch.setattr(graph, "_wrap", self._wrap)
        monkeypatch.setattr(graph, "_interview_compose_node", lambda _agent: self._compose)

    def _node(self, name: str, delta: Callable[[], dict[str, object]]) -> Callable[[SynthState], dict[str, object]]:
        def node(_state: SynthState) -> dict[str, object]:
            self.visited.append(name)
            return delta()

        return node

    def _wrap(self, _agent: Any, state_type: Any, _entry: Any, _exit: Any) -> Callable[[SynthState], Any]:
        name = state_type.__name__
        scripted: dict[str, Callable[[], dict[str, object]]] = {
            "GapDetectorState": lambda: {"gaps": self._gaps.pop(0) if self._gaps else []},
        }
        return self._node(name, scripted.get(name, dict))

    def _compose(self, _state: SynthState) -> dict[str, object]:
        self.visited.append("interview_compose")
        return {"composed_questions": [{"gap": "scope", "question": "Scope?", "options": []}]}


def _run(state: SynthState, *, resume: Any = None, checkpointer: MemorySaver | None = None) -> dict[str, Any]:
    compiled = graph.build_synth_graph(checkpointer=checkpointer)
    config: Any = {"configurable": {"thread_id": "t"}}

    async def go() -> dict[str, Any]:
        first = await compiled.ainvoke(state, config)
        if resume is None:
            return first
        assert "__interrupt__" in first, "expected the interview to pause the graph"
        return await compiled.ainvoke(Command(resume=resume), config)

    return asyncio.run(go())


def test_a_non_interactive_run_skips_the_interview_and_goes_straight_to_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = _Script(monkeypatch, gaps_per_pass=[["scope"]])  # gaps exist, but nobody can be asked
    inputs = SynthInputs(target_name="t", skip_nl_parser=True, skip_github_analysis=True, skip_api_probe=True)

    result = _run(_state(inputs=inputs, interactive=False))

    assert script.visited == [
        "ProfileSynthesizerState",
        "GapDetectorState",
        "RequestGeneratorState",
        "CriticState",
        "ProfileWriterState",
    ]
    assert result["interviewer_answers"] == []


def test_all_ingestion_sources_run_before_a_single_synthesis_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, gaps_per_pass=[[]])

    _run(_state(interactive=False))

    assert sorted(script.visited[:3]) == ["ApiProberState", "GitHubAnalyzerState", "NLParserState"]
    assert script.visited[3:5] == ["ProfileSynthesizerState", "GapDetectorState"]
    assert script.visited.count("ProfileSynthesizerState") == 1  # the fan-in joins; it does not run per source


def test_an_interactive_run_interrupts_for_answers_then_resynthesises_once(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, gaps_per_pass=[["scope"], ["scope"]])
    inputs = SynthInputs(target_name="t", skip_nl_parser=True, skip_github_analysis=True, skip_api_probe=True)

    result = _run(
        _state(inputs=inputs, interactive=True),
        resume=[{"gap": "scope", "answer": "only read-only queries"}],
        checkpointer=MemorySaver(),
    )

    assert result["interviewer_answers"] == [("scope", "Scope?", "only read-only queries")]
    assert result["interview_rounds"] == 1
    assert result["composed_questions"] == []
    assert script.visited == [
        "ProfileSynthesizerState",
        "GapDetectorState",
        "interview_compose",
        "ProfileSynthesizerState",  # re-merge with the new answers
        "GapDetectorState",
        "RequestGeneratorState",  # the gap is answered, so no second interview
        "CriticState",
        "ProfileWriterState",
    ]


def test_an_interactive_run_with_no_gaps_never_pauses(monkeypatch: pytest.MonkeyPatch) -> None:
    script = _Script(monkeypatch, gaps_per_pass=[[]])
    inputs = SynthInputs(target_name="t", skip_nl_parser=True, skip_github_analysis=True, skip_api_probe=True)

    _run(_state(inputs=inputs, interactive=True), checkpointer=MemorySaver())

    assert "interview_compose" not in script.visited


def test_unanswered_gaps_stop_after_the_round_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    # The user submits nothing each round; the loop must still end once max_interview_rounds is spent.
    script = _Script(monkeypatch, gaps_per_pass=[["scope"]] * 5)
    inputs = SynthInputs(target_name="t", skip_nl_parser=True, skip_github_analysis=True, skip_api_probe=True)
    compiled = graph.build_synth_graph(checkpointer=MemorySaver())
    config: Any = {"configurable": {"thread_id": "t"}}

    async def go() -> dict[str, Any]:
        result = await compiled.ainvoke(_state(inputs=inputs, interactive=True, max_interview_rounds=2), config)
        while "__interrupt__" in result:
            result = await compiled.ainvoke(Command(resume=[]), config)
        return result

    result = asyncio.run(go())

    assert script.visited.count("interview_compose") == 2
    assert result["interview_rounds"] == 2
    assert script.visited[-3:] == ["RequestGeneratorState", "CriticState", "ProfileWriterState"]


def test_the_compiled_graph_exposes_every_pipeline_stage() -> None:
    nodes = set(graph.build_synth_graph().get_graph().nodes)

    assert {
        "nl_parser",
        "github_analyzer",
        "api_prober",
        "profile_synthesizer",
        "gap_detector",
        "interview_compose",
        "interview_ask",
        "request_generator",
        "critic",
        "profile_writer",
    } <= nodes
