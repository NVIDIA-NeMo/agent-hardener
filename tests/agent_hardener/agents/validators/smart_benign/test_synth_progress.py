# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth progress: one line per finished node with what it produced, and a spinner paused for the interview."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import pytest

from agent_hardener.agents.validators.smart_benign import graph, progress
from agent_hardener.agents.validators.smart_benign.models import (
    GeneratedRequest,
    Persona,
    ToolSpec,
    VictimCapabilityProfile,
)
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState


def _tools(n: int) -> list[ToolSpec]:
    return [ToolSpec(name=f"t{i}", description="d", source="nl", confidence=0.9) for i in range(n)]


def _requests(n: int) -> list[GeneratedRequest]:
    return [GeneratedRequest(tool="t0", payload=f"p{i}", label="benign") for i in range(n)]


def _state(**inputs: Any) -> SynthState:
    return SynthState(inputs=SynthInputs(target_name="victim", **inputs), interactive=True)


def test_every_graph_node_has_a_label() -> None:
    nodes = set(graph.build_synth_graph().get_graph().nodes) - {"__start__", "__end__"}

    assert set(progress.PHASE_LABELS) == nodes


def test_each_finished_node_prints_what_it_produced(capsys: pytest.CaptureFixture[str]) -> None:
    profile = VictimCapabilityProfile(
        target_name="victim",
        input_hash="h",
        generated_at=datetime.now(UTC),
        tools=_tools(6),
        personas=[Persona(name="analyst", description="d")],
        out_of_scope=["a", "b"],
    )
    question = {"gap": "g1", "question": "Q?", "options": []}
    with progress.SynthProgress(_state(skip_nl_parser=True, skip_github_analysis=True)) as p:
        p.on_node(
            "api_prober",
            {"partials": {"api_probe": _tools(6)}, "source_details": {"api_probe": {"probe_errors": ["x"]}}},
        )
        p.on_node("profile_synthesizer", {"profile": profile})
        p.on_node("gap_detector", {"gaps": ["g1", "g2"]})
        p.on_node("interview_compose", {"composed_questions": [question, question]})
        p.on_node("interview_ask", {"interviewer_answers": [("g1", "Q?", "A")], "composed_questions": []})
        p.on_node("request_generator", {"requests": _requests(58)})
        p.on_node("critic", {"requests": _requests(42), "errors": ["dropped one"]})
        p.on_node("not_a_node", {})

    out = capsys.readouterr().out
    assert "probed the victim endpoint for tools — 6 tools, 1 probe error (" in out
    assert "merged the victim capability profile — 6 tools, 1 persona, 2 out-of-scope (" in out
    assert "checked the profile for gaps — 2 gaps unresolved (" in out
    assert "composed the interview questions — 2 questions (" in out
    assert "recorded the interview answers — 1 of 2 answered\n" in out  # no duration: it is the operator's time
    assert "generated the benign request suite — 58 requests (" in out
    assert "filtered and deduplicated requests — 58 → 42 kept, 1 error (" in out
    assert "not_a_node" not in out
    assert "Benign suite synthesized in 0s" in out


def test_the_spinner_names_the_node_that_runs_next(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "is_rich", lambda: True)
    monkeypatch.setattr(progress, "ok", lambda _line: None)
    with progress.SynthProgress(_state(skip_github_analysis=True)) as p:
        assert p._progress is not None
        assert p._task is not None
        task = p._progress.tasks[0]
        assert "reading the description, victim endpoint…" in task.description
        p.on_node("api_prober", {})
        assert "reading the description…" in task.description
        p.on_node("nl_parser", {})
        assert "merging the victim capability profile" in task.description
        p.on_node("gap_detector", {"gaps": ["g1"]})  # interactive with an unresolved gap → interview
        assert "composing interview questions" in task.description
        p.on_node("gap_detector", {"gaps": []})
        assert "generating the benign request suite" in task.description


def test_the_spinner_is_released_while_the_interview_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(progress, "is_rich", lambda: True)
    with progress.SynthProgress(_state()) as p:
        assert p._progress is not None
        live = p._progress.live
        seen: list[bool] = []

        async def provider(_questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
            seen.append(live.is_started)
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            asyncio.run(p.pausing(provider)([]))

        assert seen == [False]
        assert live.is_started  # restarted even though the provider raised


def test_no_total_is_printed_when_synthesis_raises(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(RuntimeError), progress.SynthProgress(_state()):
        raise RuntimeError

    assert "synthesized in" not in capsys.readouterr().out


@pytest.mark.parametrize(("seconds", "text"), [(4.4, "4s"), (59.6, "1m 00s"), (125, "2m 05s")])
def test_durations_switch_to_minutes_past_a_minute(seconds: float, text: str) -> None:
    assert progress._duration(seconds) == text
