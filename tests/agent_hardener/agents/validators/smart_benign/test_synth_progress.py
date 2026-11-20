# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth progress: one ``synth_phase`` event per finished node, with what it produced and what runs next."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from agent_hardener.agents.validators.smart_benign import graph, progress
from agent_hardener.agents.validators.smart_benign.models import (
    GeneratedRequest,
    Persona,
    ToolSpec,
    VictimCapabilityProfile,
)
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.loggers import using_event_sink


def _tools(n: int) -> list[ToolSpec]:
    return [ToolSpec(name=f"t{i}", description="d", source="nl", confidence=0.9) for i in range(n)]


def _requests(n: int) -> list[GeneratedRequest]:
    return [GeneratedRequest(tool="t0", payload=f"p{i}", label="benign") for i in range(n)]


def _state(**inputs: Any) -> SynthState:
    return SynthState(inputs=SynthInputs(target_name="victim", **inputs), interactive=True)


def _phases(tracker: progress.PhaseTracker, *steps: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    seen: list[dict[str, Any]] = []
    with using_event_sink(lambda event, payload: seen.append(payload) if event == "synth_phase" else None):
        for name, update in steps:
            tracker.on_node(name, update)
    return seen


def test_every_graph_node_has_a_label() -> None:
    nodes = set(graph.build_synth_graph().get_graph().nodes) - {"__start__", "__end__"}

    assert set(progress.PHASE_LABELS) == nodes


def test_each_event_says_what_the_node_produced() -> None:
    profile = VictimCapabilityProfile(
        target_name="victim",
        input_hash="h",
        generated_at=datetime.now(UTC),
        tools=_tools(6),
        personas=[Persona(name="analyst", description="d")],
        out_of_scope=["a", "b"],
    )
    question = {"gap": "g1", "question": "Q?", "options": []}
    tracker = progress.PhaseTracker(_state(skip_nl_parser=True, skip_github_analysis=True))

    seen = _phases(
        tracker,
        (
            "api_prober",
            {"partials": {"api_probe": _tools(6)}, "source_details": {"api_probe": {"probe_errors": ["x"]}}},
        ),
        ("profile_synthesizer", {"profile": profile}),
        ("gap_detector", {"gaps": ["g1", "g2"]}),
        ("interview_compose", {"composed_questions": [question, question]}),
        ("interview_ask", {"interviewer_answers": [("g1", "Q?", "A")], "composed_questions": []}),
        ("request_generator", {"requests": _requests(58)}),
        ("critic", {"requests": _requests(42), "errors": ["dropped one"]}),
        ("not_a_node", {}),
    )

    assert [(e["phase"], e["detail"]) for e in seen] == [
        ("api_prober", "6 tools, 1 probe error"),
        ("profile_synthesizer", "6 tools, 1 persona, 2 out-of-scope"),
        ("gap_detector", "2 gaps unresolved"),
        ("interview_compose", "2 questions"),
        ("interview_ask", "1 of 2 answered"),
        ("request_generator", "58 requests"),
        ("critic", "58 → 42 kept, 1 error"),
    ]
    assert seen[-1]["label"] == "filtered and deduplicated requests"
    assert isinstance(seen[-1]["seconds"], float)
    assert seen[4]["seconds"] is None  # the interview's time is mostly the operator's


def test_each_event_names_the_node_that_runs_next() -> None:
    tracker = progress.PhaseTracker(_state(skip_github_analysis=True))
    assert tracker.first_step() == "reading the description, victim endpoint"

    seen = _phases(
        tracker,
        ("api_prober", {}),
        ("nl_parser", {}),
        ("gap_detector", {"gaps": ["g1"]}),  # interactive with an unresolved gap → interview
        ("gap_detector", {"gaps": []}),
        ("critic", {}),
        ("profile_writer", {}),
    )

    assert [e["next"] for e in seen] == [
        "reading the description",
        "merging the victim capability profile",
        "composing interview questions",
        "generating the benign request suite",
        "writing profile.json + requests.csv",
        "finishing",
    ]
