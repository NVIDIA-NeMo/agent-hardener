# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the api_prober ``extract`` node: turning probe transcripts into capped ToolSpec hypotheses."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.nodes import extract as ex
from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.schemas import (
    API_PROBE_MAX_CONFIDENCE,
    ProberOutput,
    ProberToolHypothesis,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.api_prober.state import ApiProberState
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable

    from agent_hardener.agents.validators.smart_benign.models import ToolSpec


def _state(**overrides: Any) -> ApiProberState:
    base = {
        "target_url": "http://victim.test/v1/chat/completions",
        "target_name": "victim",
        "probe_model": "victim-model",
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
        "probe_responses": [("What can you do?", "I can search the web.")],
    }
    return ApiProberState(**{**base, **overrides})


def _hypothesis(name: str = "web_search", confidence: float = 0.4, **overrides: Any) -> ProberToolHypothesis:
    base = {"name": name, "description": f"{name} desc", "confidence": confidence}
    return ProberToolHypothesis(**{**base, **overrides})


def _patch(monkeypatch: pytest.MonkeyPatch, model: Any) -> Any:
    monkeypatch.setattr(ex, "build_chat_model", lambda **_kw: model)
    return model


def _tools(delta: dict[str, object]) -> list[ToolSpec]:
    return cast("list[ToolSpec]", delta["tools"])


def test_no_responses_skips_the_llm_and_forwards_the_probe_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ex, "build_chat_model", lambda **_kw: pytest.fail("must not call the LLM"))
    state = _state(probe_responses=[], probe_errors=["probe 1: timeout", "probe 2: HTTP 500"])

    delta = asyncio.run(ex.extract(state))

    assert delta == {
        "source_note": "api_prober: no responses, 2 error(s)",
        "errors": ["probe 1: timeout", "probe 2: HTTP 500"],
    }


def test_hypotheses_become_api_probe_tool_specs(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    output = ProberOutput(
        tools=[_hypothesis("web_search", 0.4, example_inputs=["cats"]), _hypothesis("calc", 0.1)],
        system_role="research bot",
        out_of_scope=["medical advice"],
    )
    _patch(monkeypatch, fake_model(output))

    delta = asyncio.run(ex.extract(_state(max_probes=5)))

    assert [(t.name, t.source, t.confidence) for t in _tools(delta)] == [
        ("web_search", "api_probe", 0.4),
        ("calc", "api_probe", 0.1),
    ]
    assert _tools(delta)[0].example_inputs == ["cats"]
    assert delta["system_role"] == "research bot"
    assert delta["out_of_scope"] == ["medical advice"]
    assert delta["source_note"] == "api_prober probed 1/5; extracted 2 tool(s)"


@pytest.mark.parametrize(
    ("claimed", "expected"),
    [(1.0, API_PROBE_MAX_CONFIDENCE), (0.51, API_PROBE_MAX_CONFIDENCE), (0.5, 0.5), (0.0, 0.0)],
)
def test_confidence_is_capped_because_the_victim_may_hallucinate(
    monkeypatch: pytest.MonkeyPatch, claimed: float, expected: float, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(ProberOutput(tools=[_hypothesis(confidence=claimed)])))

    delta = asyncio.run(ex.extract(_state()))

    assert _tools(delta)[0].confidence == expected


def test_probe_errors_ride_along_on_success(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(monkeypatch, fake_model(ProberOutput()))

    delta = asyncio.run(ex.extract(_state(probe_errors=["probe 2: timeout"])))

    assert delta["errors"] == ["probe 2: timeout"]
    assert delta["tools"] == []


def test_the_transcript_is_rendered_as_qa_pairs(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = _patch(monkeypatch, fake_model(ProberOutput()))
    state = _state(probe_responses=[("Q one", "A one"), ("Q two", "A two")])

    asyncio.run(ex.extract(state))

    system, user = model.messages
    assert system.content
    assert "victim" in user.content
    assert "Q: Q one\nA: A one\n\nQ: Q two\nA: A two" in user.content


def test_an_llm_failure_appends_to_the_probe_errors(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=ValueError("bad json")))

    delta = asyncio.run(ex.extract(_state(probe_errors=["probe 2: timeout"])))

    assert delta["errors"] == ["probe 2: timeout", "api_prober.extract: bad json"]
    assert "tools" not in delta
    assert "bad json" in str(delta["source_note"])


def test_a_rate_limit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(monkeypatch, fake_model(raises=RateLimitError("slow down", source="llm")))

    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(ex.extract(_state()))


def test_a_provider_rate_limit_message_is_promoted(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=RuntimeError("Too Many Requests")))

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(ex.extract(_state()))

    assert exc_info.value.source == "smart_benign:api_prober.extract"


@pytest.mark.parametrize("name", ["Web Search", "9lives", "", "has-dash"])
def test_the_schema_rejects_tool_names_that_are_not_snake_case(name: str) -> None:
    with pytest.raises(ValidationError):
        _hypothesis(name)
