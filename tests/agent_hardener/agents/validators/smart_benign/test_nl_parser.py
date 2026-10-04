# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the nl_parser subgraph (description → typed capabilities, one LLM call).

The LLM is the canonical ``_FakeModel`` / ``build_chat_model`` monkeypatch — no network.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from agent_hardener.agents.validators.smart_benign.subgraphs.nl_parser.nodes import parse as pr
from agent_hardener.agents.validators.smart_benign.subgraphs.nl_parser.schemas import (
    ParserOutput,
    ParserPersona,
    ParserToolHypothesis,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.nl_parser.state import NLParserState
from agent_hardener.rate_limits import RateLimitError


def _state(**overrides: Any) -> NLParserState:
    base = {
        "description": "A research assistant with a bash_executor tool.",
        "target_name": "victim",
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return NLParserState(**{**base, **overrides})


class _FakeModel:
    def __init__(self, output: ParserOutput | None = None, raises: Exception | None = None) -> None:
        self._output = output
        self._raises = raises

    def with_structured_output(self, _schema: object) -> _FakeModel:
        return self

    async def ainvoke(self, _messages: list[object]) -> ParserOutput:
        if self._raises is not None:
            raise self._raises
        assert self._output is not None
        return self._output


def _patch(monkeypatch: pytest.MonkeyPatch, model: _FakeModel) -> None:
    monkeypatch.setattr(pr, "build_chat_model", lambda **_kw: model)


def test_parse_skips_without_description() -> None:
    delta = asyncio.run(pr.parse(_state(description=None)))
    assert "skipped" in delta["source_note"]
    assert "tools" not in delta  # nothing parsed


def test_parse_maps_llm_output_to_nl_tagged_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(
        monkeypatch,
        _FakeModel(
            ParserOutput(
                system_role="research assistant",
                tools=[ParserToolHypothesis(name="bash_executor", description="runs bash", confidence=0.8)],
                personas=[ParserPersona(name="dev", description="a developer")],
                out_of_scope=["deleting files"],
            )
        ),
    )
    delta = asyncio.run(pr.parse(_state()))

    assert [(t.name, t.source) for t in delta["tools"]] == [("bash_executor", "nl")]
    assert [p.name for p in delta["personas"]] == ["dev"]
    assert delta["system_role"] == "research assistant"
    assert delta["out_of_scope"] == ["deleting files"]
    assert "1 tool" in delta["source_note"]


def test_parse_rate_limit_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeModel(raises=Exception("429 too many requests")))
    with pytest.raises(RateLimitError):
        asyncio.run(pr.parse(_state()))


def test_parse_swallows_non_rate_limit_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, _FakeModel(raises=ValueError("bad json")))
    delta = asyncio.run(pr.parse(_state()))
    assert delta["errors"]
    assert "nl_parser" in delta["errors"][0]
    assert "tools" not in delta  # a failed parse contributes no tools
