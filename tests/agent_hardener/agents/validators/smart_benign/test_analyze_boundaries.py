# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gap_detector/nodes/analyze_boundaries.py."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.nodes import analyze_boundaries as ab
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.schemas import BoundaryVerdict
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.state import GapDetectorState
from agent_hardener.rate_limits import RateLimitError

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _tool(name: str) -> ToolSpec:
    return ToolSpec(name=name, description="does things", source="nl", confidence=0.9, example_inputs=["hi"])


def _profile(tools: list[ToolSpec]) -> VictimCapabilityProfile:
    return VictimCapabilityProfile(
        target_name="test-agent",
        system_role="role",
        tools=tools,
        personas=[],
        out_of_scope=[],
        input_hash="abc123",
        generated_at=_NOW,
    )


def _state(**kwargs: object) -> GapDetectorState:
    base: dict[str, object] = {
        "gaps": ["rule gap"],
        "interview_open": True,
        "synth_model": "m",
        "synth_base_url": "u",
        "synth_api_key": "k",
    }
    base.update(kwargs)
    return GapDetectorState(**base)


def test_closed_interview_makes_no_llm_call(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def _boom(_tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        nonlocal called
        called = True
        raise AssertionError("should not be called")

    monkeypatch.setattr(ab, "_call", _boom)
    # interview_open=False (budget spent / non-interactive) → boundary fan-out skipped.
    state = _state(interview_open=False, profile=_profile([_tool("bash_executor")]))
    result = asyncio.run(ab.analyze_boundaries(state))
    assert result == {}
    assert called is False


def test_emits_boundary_gap_with_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _call(tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        return BoundaryVerdict(needs_clarification=True, question=f"is X ok for {tool.name}?")

    monkeypatch.setattr(ab, "_call", _call)
    state = _state(profile=_profile([_tool("bash_executor")]))
    result = asyncio.run(ab.analyze_boundaries(state))
    gaps = result["gaps"]
    assert gaps[0] == "rule gap"  # rule gaps preserved, boundary appended
    assert gaps[1] == "boundary[bash_executor]: is X ok for bash_executor?"


def test_skips_tool_when_verdict_clear(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _call(_tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        return BoundaryVerdict(needs_clarification=False)

    monkeypatch.setattr(ab, "_call", _call)
    state = _state(profile=_profile([_tool("bash_executor")]))
    result = asyncio.run(ab.analyze_boundaries(state))
    assert result == {"gaps": ["rule gap"]}


def test_skips_already_answered_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    async def _call(tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        calls.append(tool.name)
        return BoundaryVerdict(needs_clarification=True, question="q")

    monkeypatch.setattr(ab, "_call", _call)
    prior = [("boundary[bash_executor]: old question", "Q?", "only relative paths")]
    state = _state(
        profile=_profile([_tool("bash_executor"), _tool("file_reader")]),
        interviewer_answers=prior,
    )
    result = asyncio.run(ab.analyze_boundaries(state))
    assert calls == ["file_reader"]  # bash_executor already answered → skipped
    assert result["gaps"] == ["rule gap", "boundary[file_reader]: q"]


def test_no_tools_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "_call", lambda *_: pytest.fail("unreachable"))
    state = _state(profile=_profile([]))
    assert asyncio.run(ab.analyze_boundaries(state)) == {}


def test_all_tools_answered_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "_call", lambda *_: pytest.fail("unreachable"))
    prior = [("boundary[bash_executor]: q", "Q?", "a")]
    state = _state(profile=_profile([_tool("bash_executor")]), interviewer_answers=prior)
    assert asyncio.run(ab.analyze_boundaries(state)) == {}


def test_tool_failure_is_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _call(_tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        raise ValueError("model exploded")

    monkeypatch.setattr(ab, "_call", _call)
    state = _state(profile=_profile([_tool("bash_executor")]))
    result = asyncio.run(ab.analyze_boundaries(state))
    assert result == {"gaps": ["rule gap"]}  # failed tool drops out, no crash


def test_rate_limit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _call(_tool: ToolSpec, _state: GapDetectorState) -> BoundaryVerdict:
        raise RateLimitError("slow down", source="test")

    monkeypatch.setattr(ab, "_call", _call)
    state = _state(profile=_profile([_tool("bash_executor")]))
    with pytest.raises(RateLimitError):
        asyncio.run(ab.analyze_boundaries(state))


def test_call_invokes_model_and_builds_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeModel:
        def with_structured_output(self, _schema: object) -> _FakeModel:
            return self

        async def ainvoke(self, messages: list[object]) -> BoundaryVerdict:
            captured["messages"] = messages
            return BoundaryVerdict(needs_clarification=True, question="absolute paths?")

    monkeypatch.setattr(ab, "build_chat_model", lambda **_kw: _FakeModel())
    state = _state(profile=_profile([_tool("bash_executor")]))
    result = asyncio.run(ab.analyze_boundaries(state))
    assert result["gaps"] == ["rule gap", "boundary[bash_executor]: absolute paths?"]
    # system + user message reached the model
    assert len(captured["messages"]) == 2


def _fake_model_raising(exc: Exception) -> object:
    class _FakeModel:
        def with_structured_output(self, _schema: object) -> object:
            return self

        async def ainvoke(self, _messages: list[object]) -> object:
            raise exc

    return _FakeModel()


def test_call_non_rate_limit_error_drops_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "build_chat_model", lambda **_kw: _fake_model_raising(ValueError("bad json")))
    state = _state(profile=_profile([_tool("bash_executor")]))
    # _call re-raises the plain error; _one swallows it → only rule gaps remain.
    assert asyncio.run(ab.analyze_boundaries(state)) == {"gaps": ["rule gap"]}


def test_call_maps_rate_limit_and_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ab, "build_chat_model", lambda **_kw: _fake_model_raising(Exception("429 too many requests")))
    state = _state(profile=_profile([_tool("bash_executor")]))
    with pytest.raises(RateLimitError):
        asyncio.run(ab.analyze_boundaries(state))
