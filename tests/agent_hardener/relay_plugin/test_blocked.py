# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shape of the canned answer, tested without a Relay runtime or an agent framework.

The module under test imports no ``nemo_relay``, which is the point: picking the wrong payload shape
is what breaks a victim, so it should be checkable in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_hardener.relay_plugin import blocked
from agent_hardener.relay_plugin.config import DEFAULT_BLOCKED_MESSAGE

pytestmark = pytest.mark.unit

_MESSAGE = DEFAULT_BLOCKED_MESSAGE


class _Request:
    """Stands in for ``nemo_relay.LLMRequest``, which is native and cannot be built cheaply."""

    def __init__(self, content: Any) -> None:
        self.content = content


def _langchain(messages: list[Any]) -> _Request:
    return _Request({"messages": messages, "model": "m"})


def test_a_langchain_request_gets_relays_model_response_envelope() -> None:
    """Relay's LangChain decoder raises on any other shape, so this key is load-bearing."""
    payload = blocked.blocked_response(_langchain([{"type": "human", "data": {"content": "hi"}}]), _MESSAGE)

    assert payload is not None
    envelope = payload[blocked.model_response_key()]
    assert envelope["messages"][0]["data"]["content"] == _MESSAGE
    # No structured_response: Relay only decodes the key when present, and a None would round-trip
    # through the codec for nothing.
    assert "structured_response" not in envelope


def test_an_openai_request_gets_a_chat_completion() -> None:
    payload = blocked.blocked_response(_Request({"messages": [{"role": "user", "content": "hi"}]}), _MESSAGE)

    assert payload is not None
    assert payload["choices"][0]["message"] == {"role": "assistant", "content": _MESSAGE, "tool_calls": []}
    assert payload["choices"][0]["finish_reason"] == "stop"


def test_the_answer_carries_no_tool_calls() -> None:
    """The empty tool call list is what ends the agent loop; Relay has no abort-turn API."""
    assert blocked.ai_message_dicts(_MESSAGE)[0]["data"]["tool_calls"] == []


@pytest.mark.parametrize(
    "content",
    [
        {"messages": ["not-a-mapping"]},
        {"messages": [{"neither": "role nor type"}]},
        {"messages": []},
        {"messages": "not-a-list"},
        {"no-messages-at-all": True},
        "not-a-mapping",
        None,
    ],
    ids=[
        "non-mapping-entry",
        "unknown-keys",
        "empty",
        "messages-not-a-list",
        "no-messages",
        "content-not-a-map",
        "none",
    ],
)
def test_an_unrecognised_request_fails_open(content: Any) -> None:
    """``None`` means "let the model run" — a guess would raise through the whole turn."""
    assert blocked.blocked_response(_Request(content), _MESSAGE) is None


def test_a_request_object_without_content_fails_open() -> None:
    assert blocked.blocked_response(object(), _MESSAGE) is None


def test_a_type_key_without_a_data_mapping_is_not_langchain() -> None:
    """``messages_to_dict`` always pairs ``type`` with a ``data`` mapping; anything else is foreign."""
    assert blocked.blocked_response(_langchain([{"type": "human", "data": "not-a-mapping"}]), _MESSAGE) is None


def test_stream_chunks_end_the_stream() -> None:
    chunks = blocked.blocked_stream_chunks(_Request({"messages": [{"role": "user", "content": "hi"}]}), _MESSAGE)

    assert chunks is not None
    assert chunks[0]["choices"][0]["delta"]["content"] == _MESSAGE
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


def test_streaming_only_shapes_the_openai_form() -> None:
    """No shipped Relay integration streams, so there is no LangChain chunk format to copy.

    Guessing one would be fed to the victim's own collector and finalizer — a worse failure than
    letting the call through.
    """
    assert blocked.blocked_stream_chunks(_langchain([{"type": "human", "data": {"content": "hi"}}]), _MESSAGE) is None


def test_the_envelope_key_falls_back_when_langchain_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The plugin is staged into victims with no LangChain at all, where the import raises."""
    blocked.model_response_key.cache_clear()
    monkeypatch.setitem(__import__("sys").modules, "nemo_relay.integrations.langchain._serialization", None)
    try:
        assert blocked.model_response_key() == "__nemo_relay_integrations_langchain_model_response"
    finally:
        blocked.model_response_key.cache_clear()


def test_the_message_dicts_survive_without_langchain_core(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermes and friends have no langchain_core; the literal fallback must still be usable."""
    monkeypatch.setitem(__import__("sys").modules, "langchain_core.messages", None)
    dicts = blocked.ai_message_dicts(_MESSAGE)

    assert dicts == [{"type": "ai", "data": {"content": _MESSAGE, "type": "ai", "tool_calls": []}}]


def test_the_shipped_message_is_the_one_the_user_sees() -> None:
    """Locks the exact wording: it is the entire answer a blocked request gets, so drift is visible.

    Asserted literally rather than against the constant — a test that reads the constant back would
    pass no matter what the constant became.
    """
    assert DEFAULT_BLOCKED_MESSAGE == "The request was blocked by security policy."


def test_the_message_reaches_both_harness_shapes() -> None:
    """Whatever the harness, the answer carries that exact text and nothing else."""
    langchain = blocked.blocked_response(_langchain([{"type": "human", "data": {"content": "hi"}}]), _MESSAGE)
    openai = blocked.blocked_response(_Request({"messages": [{"role": "user", "content": "hi"}]}), _MESSAGE)
    assert DEFAULT_BLOCKED_MESSAGE in str(langchain)
    assert openai["choices"][0]["message"]["content"] == DEFAULT_BLOCKED_MESSAGE
