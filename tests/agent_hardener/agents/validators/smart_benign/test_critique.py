# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the critic subgraph: LLM drop/rewrite decisions plus the deterministic dedup safety net."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, cast

import pytest

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, RequestLabel
from agent_hardener.agents.validators.smart_benign.subgraphs.critic.nodes import critique as cr
from agent_hardener.agents.validators.smart_benign.subgraphs.critic.schemas import CriticDecision, CriticResult
from agent_hardener.agents.validators.smart_benign.subgraphs.critic.state import CriticState
from agent_hardener.rate_limits import RateLimitError

if TYPE_CHECKING:
    from collections.abc import Callable


def _request(tool: str, payload: str, label: RequestLabel = "benign") -> GeneratedRequest:
    return GeneratedRequest(tool=tool, payload=payload, label=label)


def _decision(index: int, action: Any = "keep", new_payload: str | None = None) -> CriticDecision:
    return CriticDecision(index=index, action=action, new_payload=new_payload, rationale="because")


def _state(requests: list[GeneratedRequest], **overrides: Any) -> CriticState:
    base = {
        "target_name": "victim",
        "requests": requests,
        "synth_model": "m",
        "synth_base_url": "http://llm.test",
        "synth_api_key": "k",
    }
    return CriticState(**{**base, **overrides})


def _patch(monkeypatch: pytest.MonkeyPatch, model: Any) -> None:
    monkeypatch.setattr(cr, "build_chat_model", lambda **_kw: model)


def _filtered(delta: dict[str, object]) -> list[GeneratedRequest]:
    return cast("list[GeneratedRequest]", delta["filtered_requests"])


def _payloads(delta: dict[str, object]) -> list[str]:
    return [r.payload for r in _filtered(delta)]


def test_empty_suite_short_circuits_without_calling_the_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cr, "build_chat_model", lambda **_kw: pytest.fail("must not call the LLM"))

    delta = asyncio.run(cr.critique(_state([])))

    assert delta == {"filtered_requests": [], "source_note": "no requests to critique"}


def test_decisions_drop_rewrite_and_keep_rows(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    requests = [_request("bash", "ls"), _request("bash", "rm -rf /"), _request("search", "cats")]
    result = CriticResult(
        decisions=[_decision(1), _decision(2, "drop"), _decision(3, "rewrite", "kittens, please")],
    )
    model = fake_model(result)
    _patch(monkeypatch, model)

    delta = asyncio.run(cr.critique(_state(requests)))

    assert _payloads(delta) == ["ls", "kittens, please"]
    assert _filtered(delta)[1].tool == "search"
    assert delta["source_note"] == "critic kept 2/3 (LLM); 2 after exact dedup"
    assert "errors" not in delta
    assert model.schema is CriticResult


def test_the_prompt_carries_the_target_tools_scope_and_numbered_rows(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    model = fake_model(CriticResult(decisions=[_decision(1)]))
    _patch(monkeypatch, model)
    state = _state(
        [_request("bash", "ls -la", "borderline_benign")], tool_names=["bash", "search"], out_of_scope=["no deletes"]
    )

    asyncio.run(cr.critique(state))

    system, user = model.messages
    assert system.content
    assert "victim" in user.content
    assert "- bash\n- search" in user.content
    assert "- no deletes" in user.content
    assert "[1] tool=bash label=borderline_benign payload='ls -la'" in user.content


def test_exact_duplicates_are_removed_even_when_the_llm_kept_them(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    requests = [_request("bash", "ls"), _request("bash", "ls"), _request("search", "ls")]
    _patch(monkeypatch, fake_model(CriticResult(decisions=[_decision(i) for i in (1, 2, 3)])))

    delta = asyncio.run(cr.critique(_state(requests)))

    assert [(r.tool, r.payload) for r in _filtered(delta)] == [("bash", "ls"), ("search", "ls")]
    assert delta["source_note"] == "critic kept 3/3 (LLM); 2 after exact dedup"


def test_rows_the_critic_forgot_are_kept(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fake_model: Callable[..., Any]
) -> None:
    requests = [_request("a", "one"), _request("a", "two")]
    _patch(monkeypatch, fake_model(CriticResult(decisions=[_decision(1)])))

    with caplog.at_level(logging.WARNING, logger=cr.logger.name):
        delta = asyncio.run(cr.critique(_state(requests)))

    assert _payloads(delta) == ["one", "two"]
    assert "critic missed index 2" in caplog.text


def test_the_last_decision_for_an_index_wins(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, fake_model: Callable[..., Any]
) -> None:
    result = CriticResult(decisions=[_decision(1, "drop"), _decision(1, "keep")])
    _patch(monkeypatch, fake_model(result))

    with caplog.at_level(logging.WARNING, logger=cr.logger.name):
        delta = asyncio.run(cr.critique(_state([_request("a", "one")])))

    assert _payloads(delta) == ["one"]
    assert "multiple decisions for index 1" in caplog.text


@pytest.mark.parametrize("new_payload", [None, ""])
def test_a_rewrite_without_a_payload_keeps_the_original(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    new_payload: str | None,
    fake_model: Callable[..., Any],
) -> None:
    _patch(monkeypatch, fake_model(CriticResult(decisions=[_decision(1, "rewrite", new_payload)])))

    with caplog.at_level(logging.WARNING, logger=cr.logger.name):
        delta = asyncio.run(cr.critique(_state([_request("a", "original")])))

    assert _payloads(delta) == ["original"]
    assert "without new_payload" in caplog.text


def test_a_rewrite_keeps_every_other_field_of_the_row(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    row = GeneratedRequest(tool="bash", payload="old", label="negative_control", rationale="why", persona="dev")
    _patch(monkeypatch, fake_model(CriticResult(decisions=[_decision(1, "rewrite", "new")])))

    (rewritten,) = _filtered(asyncio.run(cr.critique(_state([row]))))

    assert rewritten == row.model_copy(update={"payload": "new"})


def test_an_llm_failure_falls_back_to_exact_dedup_and_records_the_error(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    requests = [_request("a", "x"), _request("a", "x"), _request("a", "y")]
    _patch(monkeypatch, fake_model(raises=ValueError("bad structured output")))

    delta = asyncio.run(cr.critique(_state(requests)))

    assert _payloads(delta) == ["x", "y"]
    assert delta["errors"] == ["critic: bad structured output"]
    assert "critic LLM failed (bad structured output); deterministic dedup only: 3 → 2" in str(delta["source_note"])


def test_a_rate_limit_is_not_swallowed(monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]) -> None:
    _patch(monkeypatch, fake_model(raises=RateLimitError("slow down", source="llm")))

    with pytest.raises(RateLimitError, match="slow down"):
        asyncio.run(cr.critique(_state([_request("a", "x")])))


def test_a_provider_rate_limit_message_is_promoted_to_a_rate_limit_error(
    monkeypatch: pytest.MonkeyPatch, fake_model: Callable[..., Any]
) -> None:
    _patch(monkeypatch, fake_model(raises=RuntimeError("Too Many Requests, try again later")))

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(cr.critique(_state([_request("a", "x")])))

    assert exc_info.value.source == "smart_benign:critic"
    assert isinstance(exc_info.value.__cause__, RuntimeError)
