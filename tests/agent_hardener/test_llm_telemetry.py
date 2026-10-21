# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult

from agent_hardener import llm_telemetry
from agent_hardener.events import EventType
from agent_hardener.llm_telemetry import LlmTelemetryCallback
from agent_hardener.loggers import using_current_agent

_AGENT = {"agent_id": "a1", "agent_name": "Agent One", "agent_role": "defender", "validator_kind": None}


@pytest.fixture
def emitted(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    calls: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(llm_telemetry, "emit_event", lambda event, payload: calls.append((event, payload)))
    return calls


def _chat_result(content: str, **llm_output: Any) -> LLMResult:
    return LLMResult(generations=[[ChatGeneration(message=AIMessage(content=content))]], llm_output=llm_output or None)


def test_chat_call_pairs_the_prompt_with_the_completion_and_the_model(
    emitted: list[tuple[str, dict[str, Any]]],
) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()
    messages = [[SystemMessage(content="be brief"), HumanMessage(content="hi")]]

    with using_current_agent(_AGENT):
        callback.on_chat_model_start({}, messages, run_id=run_id)
        callback.on_llm_end(_chat_result("hello", model_name="gpt-x"), run_id=run_id)

    ((event, payload),) = emitted
    assert event == EventType.LLM_CALL
    assert payload == {
        "agent_id": "a1",
        "agent_name": "Agent One",
        "agent_role": "defender",
        "validator_kind": None,
        "request": "system: be brief\nhuman: hi",
        "response": "hello",
        "label": "gpt-x",
        "ok": True,
    }


def test_completion_llm_call_joins_prompts_and_reads_plain_generation_text(
    emitted: list[tuple[str, dict[str, Any]]],
) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()
    result = LLMResult(generations=[[Generation(text="done")]])

    with using_current_agent(_AGENT):
        callback.on_llm_start({}, ["first", "second"], run_id=run_id)
        callback.on_llm_end(result, run_id=run_id)

    ((_, payload),) = emitted
    assert (payload["request"], payload["response"], payload["label"]) == ("first\n\nsecond", "done", "")


def test_chat_start_with_no_messages_records_an_empty_request(emitted: list[tuple[str, dict[str, Any]]]) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()

    with using_current_agent(_AGENT):
        callback.on_chat_model_start({}, [], run_id=run_id)
        callback.on_llm_end(_chat_result("x"), run_id=run_id)

    assert emitted[0][1]["request"] == ""


def test_result_without_generations_emits_an_empty_response(emitted: list[tuple[str, dict[str, Any]]]) -> None:
    callback = LlmTelemetryCallback()

    with using_current_agent(_AGENT):
        callback.on_llm_end(LLMResult(generations=[]), run_id=uuid4())
        callback.on_llm_end(LLMResult(generations=[[]]), run_id=uuid4())

    assert [payload["response"] for _, payload in emitted] == ["", ""]


def test_error_is_reported_as_a_failed_call_carrying_the_prompt(emitted: list[tuple[str, dict[str, Any]]]) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()

    with using_current_agent(_AGENT):
        callback.on_llm_start({}, ["prompt"], run_id=run_id)
        callback.on_llm_error(TimeoutError("upstream timed out"), run_id=run_id)

    ((_, payload),) = emitted
    assert (payload["ok"], payload["label"]) == (False, "error")
    assert (payload["request"], payload["response"]) == ("prompt", "upstream timed out")


def test_unattributed_calls_emit_nothing_and_do_not_leak_the_prompt(
    emitted: list[tuple[str, dict[str, Any]]],
) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()

    callback.on_llm_start({}, ["prompt"], run_id=run_id)
    callback.on_llm_end(_chat_result("x"), run_id=run_id)
    callback.on_llm_start({}, ["prompt"], run_id=run_id)
    callback.on_llm_error(RuntimeError("boom"), run_id=run_id)

    assert emitted == []

    with using_current_agent(_AGENT):
        callback.on_llm_end(_chat_result("x"), run_id=run_id)

    ((_, payload),) = emitted
    assert payload["request"] == ""


def test_long_text_is_clipped_with_an_ellipsis_and_whitespace_is_trimmed(
    emitted: list[tuple[str, dict[str, Any]]],
) -> None:
    callback, run_id = LlmTelemetryCallback(), uuid4()
    limit = llm_telemetry._MAX_CHARS

    with using_current_agent(_AGENT):
        callback.on_llm_end(_chat_result("  " + "x" * (limit + 50) + "  "), run_id=run_id)

    response = emitted[0][1]["response"]
    assert response == "x" * limit + "…"


def test_emit_agent_exchange_is_a_noop_without_an_ambient_agent(emitted: list[tuple[str, dict[str, Any]]]) -> None:
    llm_telemetry.emit_agent_exchange(request="r", response="s", label="l", ok=True)

    assert emitted == []


@pytest.mark.parametrize("blocked", [None, True, False])
def test_emit_agent_exchange_includes_blocked_only_when_given(
    emitted: list[tuple[str, dict[str, Any]]], blocked: bool | None
) -> None:
    with using_current_agent({**_AGENT, "validator_kind": "attack"}):
        llm_telemetry.emit_agent_exchange(request="r", response="s", label="l", ok=False, blocked=blocked)

    ((event, payload),) = emitted
    assert event == EventType.AGENT_EXCHANGE
    assert payload["phase"] == "validators"
    assert payload["validator_kind"] == "attack"
    assert ("blocked" in payload) is (blocked is not None)
    assert payload.get("blocked") is blocked
