# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the request_generator subgraph: per-tool fan-out, dedup, and failure isolation."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, Persona, ToolSpec
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.nodes import generate as gen
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.schemas import (
    GeneratedRow,
    ToolGenerationOutput,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.state import RequestGeneratorState
from agent_hardener.rate_limits import RateLimitError


def _tool(name: str, **overrides: Any) -> ToolSpec:
    base = {"name": name, "description": f"{name} tool", "source": "nl", "confidence": 0.9}
    return ToolSpec(**{**base, **overrides})


def _row(payload: str, label: Any = "benign", persona: str | None = None) -> GeneratedRow:
    return GeneratedRow(payload=payload, label=label, rationale="why", persona=persona)


def _state(*tools: ToolSpec, **overrides: Any) -> RequestGeneratorState:
    base = {
        "target_name": "victim",
        "tools": list(tools),
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return RequestGeneratorState(**{**base, **overrides})


class _FakeModel:
    """Answers per tool, keyed by the tool name that appears in the rendered user prompt."""

    def __init__(self, outputs: dict[str, ToolGenerationOutput | Exception]) -> None:
        self.outputs = outputs
        self.prompts: dict[str, str] = {}

    def with_structured_output(self, schema: object) -> Any:
        assert schema is ToolGenerationOutput
        return self

    async def ainvoke(self, messages: list[Any]) -> ToolGenerationOutput:
        user = messages[1].content
        for name, output in self.outputs.items():
            if name in user:
                self.prompts[name] = user
                if isinstance(output, Exception):
                    raise output
                return output
        raise AssertionError(f"no scripted output for prompt: {user[:200]}")


def _generated(delta: dict[str, object]) -> list[GeneratedRequest]:
    return cast("list[GeneratedRequest]", delta["generated"])


def _patch(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(gen, "build_chat_model", lambda **_kw: model)


def test_no_tools_short_circuits_without_calling_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gen, "build_chat_model", lambda **_kw: pytest.fail("must not call the LLM"))

    delta = asyncio.run(gen.generate(_state()))

    assert delta == {"generated": [], "source_note": "no tools in profile; nothing to generate"}


def test_rows_are_tagged_with_their_tool_and_returned_in_tool_order(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(
        {
            "alpha_tool": ToolGenerationOutput(rows=[_row("a1"), _row("a2", "borderline_benign", persona="dev")]),
            "beta_tool": ToolGenerationOutput(rows=[_row("b1", "negative_control")]),
        }
    )
    _patch(monkeypatch, model)

    delta = asyncio.run(gen.generate(_state(_tool("alpha_tool"), _tool("beta_tool"))))

    rows = _generated(delta)
    assert [(r.tool, r.payload, r.label, r.persona) for r in rows] == [
        ("alpha_tool", "a1", "benign", None),
        ("alpha_tool", "a2", "borderline_benign", "dev"),
        ("beta_tool", "b1", "negative_control", None),
    ]
    assert delta["errors"] == []
    assert delta["source_note"] == "generated 3 row(s) across 2 tool(s)"


def test_duplicates_within_a_tool_collapse_ignoring_case_and_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    output = ToolGenerationOutput(rows=[_row("List  Files"), _row("list files"), _row(" list\tfiles\n"), _row("other")])
    _patch(monkeypatch, _FakeModel({"alpha_tool": output}))

    delta = asyncio.run(gen.generate(_state(_tool("alpha_tool"))))

    assert [r.payload for r in _generated(delta)] == ["List  Files", "other"]


def test_the_same_payload_is_kept_across_different_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    same = ToolGenerationOutput(rows=[_row("hello")])
    _patch(monkeypatch, _FakeModel({"alpha_tool": same, "beta_tool": same}))

    delta = asyncio.run(gen.generate(_state(_tool("alpha_tool"), _tool("beta_tool"))))

    assert [r.tool for r in _generated(delta)] == ["alpha_tool", "beta_tool"]


def test_one_failing_tool_does_not_sink_the_others(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(
        {
            "alpha_tool": RuntimeError("bad json"),
            "beta_tool": ToolGenerationOutput(rows=[_row("b1")]),
        }
    )
    _patch(monkeypatch, model)

    delta = asyncio.run(gen.generate(_state(_tool("alpha_tool"), _tool("beta_tool"))))

    assert [r.tool for r in _generated(delta)] == ["beta_tool"]
    assert delta["errors"] == ["request_generator[alpha_tool]: bad json"]
    assert delta["source_note"] == "generated 1 row(s) across 2 tool(s); 1 tool(s) failed"


def test_a_tool_with_no_rows_yields_nothing_and_no_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeModel({"alpha_tool": ToolGenerationOutput()}))

    delta = asyncio.run(gen.generate(_state(_tool("alpha_tool"))))

    assert delta["generated"] == []
    assert delta["errors"] == []


def test_a_rate_limit_aborts_the_whole_fan_out(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel(
        {"alpha_tool": RateLimitError("slow down", source="llm"), "beta_tool": ToolGenerationOutput(rows=[_row("b")])}
    )
    _patch(monkeypatch, model)

    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(gen.generate(_state(_tool("alpha_tool"), _tool("beta_tool"))))


def test_a_provider_rate_limit_message_is_promoted_with_the_tool_in_the_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch(monkeypatch, _FakeModel({"alpha_tool": RuntimeError("429 Too Many Requests")}))

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(gen.generate(_state(_tool("alpha_tool"))))

    assert exc_info.value.source == "smart_benign:request_generator[alpha_tool]"
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_the_prompt_carries_the_tool_profile_and_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _FakeModel({"alpha_tool": ToolGenerationOutput()})
    _patch(monkeypatch, model)
    tool = _tool(
        "alpha_tool",
        example_inputs=["ex-input-1"],
        allowed_patterns=["allow-pattern"],
        blocked_patterns=["block-pattern"],
    )
    state = _state(
        tool,
        system_role="support bot",
        personas=[Persona(name="Dana", description="a nurse")],
        out_of_scope=["legal advice"],
        requests_per_tool=7,
        borderline_per_tool=4,
        negative_control_per_tool=2,
    )

    asyncio.run(gen.generate(state))

    prompt = model.prompts["alpha_tool"]
    for expected in ("victim", "ex-input-1", "allow-pattern", "block-pattern", "support bot", "Dana", "legal advice"):
        assert expected in prompt
    assert "up to **7**" in prompt
    assert "up to **4**" in prompt
    assert "up to **2**" in prompt
