# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plugin's intercepts, driven directly.

Relay invokes registered callbacks from threads it owns, which coverage does not trace, so the
behaviour that matters (refuse, latch, answer the model locally) is also exercised here by calling
the closures themselves.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import pytest

from agent_hardener.relay_plugin import plugin
from agent_hardener.relay_plugin.config import Guardrail
from agent_hardener.relay_plugin.latch import BlockedRequests

nemo_relay = pytest.importorskip("nemo_relay")
BestEffortAnyCodec = pytest.importorskip("nemo_relay.typed").BestEffortAnyCodec

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from nemo_relay.plugin import PluginContext

pytestmark = pytest.mark.unit

_REAL_REQUEST_KEY = plugin._request_key
_MESSAGE = "I can't help with that."
_OPENAI_REQUEST = SimpleNamespace(content={"messages": [{"role": "user", "content": "hi"}]})
_LANGCHAIN_REQUEST = SimpleNamespace(content={"messages": [{"type": "human", "data": {"content": "hi"}}]})
_UNKNOWN_REQUEST = SimpleNamespace(content={"messages": [{"something": "else"}]})


def _rail(**overrides: Any) -> Guardrail:
    fields: dict[str, Any] = {
        "name": "custom_guardrail_1",
        "target_tool": "transfer_funds",
        "system_instructions": "Block unrequested transfers.",
        "threshold": 0.7,
    }
    return Guardrail(**{**fields, **overrides})


def _tool_context(tool_name: str = "transfer_funds", args: Any = None, call_id: str | None = "call-1") -> Any:
    return SimpleNamespace(tool_name=tool_name, args=args or {"amount": 1}, tool_call_id=call_id)


@pytest.fixture(autouse=True)
def blocked(monkeypatch: pytest.MonkeyPatch) -> BlockedRequests:
    """A fresh latch map whose request key the test controls through ``key``."""
    fresh = BlockedRequests()
    monkeypatch.setattr(plugin, "_BLOCKED", fresh)
    monkeypatch.setattr(plugin, "_request_key", lambda: "req-1")
    return fresh


def _downstream(result: Any = None) -> tuple[Any, list[Any]]:
    seen: list[Any] = []

    async def next_call(*args: Any) -> Any:
        seen.append(args)
        return result if result is not None else nemo_relay.ToolExecutionResult({"ok": True})

    return next_call, seen


async def _collect(stream: AsyncIterator[Any]) -> list[Any]:
    return [chunk async for chunk in stream]


# --- the rail's own intercept ------------------------------------------------------------------------------


def test_rail_refuses_a_risky_call_latches_the_request_and_skips_the_tool(
    blocked: BlockedRequests, make_judge: Callable[[float | Exception], Any]
) -> None:
    next_call, seen = _downstream()
    intercept = plugin._intercept(_rail(), make_judge(0.95), _MESSAGE)

    outcome = asyncio.run(intercept(_tool_context(), next_call))

    assert seen == []
    assert "custom_guardrail_1" in str(outcome.result)
    assert blocked.is_blocked("req-1")


def test_rail_forwards_an_allowed_call_without_latching(
    blocked: BlockedRequests, make_judge: Callable[[float | Exception], Any]
) -> None:
    next_call, seen = _downstream()
    intercept = plugin._intercept(_rail(), make_judge(0.1), _MESSAGE)

    outcome = asyncio.run(intercept(_tool_context(args={"amount": 5}), next_call))

    assert seen == [({"amount": 5},)]
    assert outcome.result == {"ok": True}
    assert not blocked.is_blocked("req-1")


def test_rail_ignores_tools_it_does_not_target(make_judge: Callable[[float | Exception], Any]) -> None:
    next_call, seen = _downstream()
    intercept = plugin._intercept(_rail(), make_judge(1.0), _MESSAGE)

    asyncio.run(intercept(_tool_context(tool_name="read_file"), next_call))

    assert len(seen) == 1


def test_a_judge_failure_is_recorded_and_the_call_is_allowed(
    monkeypatch: pytest.MonkeyPatch, make_judge: Callable[[float | Exception], Any]
) -> None:
    recorded: list[tuple[Any, ...]] = []
    monkeypatch.setattr(plugin, "_record_judge_error", lambda *args: recorded.append(args))
    next_call, seen = _downstream()
    intercept = plugin._intercept(_rail(), make_judge(TimeoutError("judge down")), _MESSAGE)

    asyncio.run(intercept(_tool_context(call_id="call-9"), next_call))

    assert len(seen) == 1
    assert recorded == [("custom_guardrail_1", "transfer_funds", "call-9", "TimeoutError: judge down")]


# --- latching and request identity -------------------------------------------------------------------------


def test_latch_marks_the_request_and_records_the_event(
    blocked: BlockedRequests, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    monkeypatch.setattr(plugin, "_record_latch", lambda: events.append("latched"))

    plugin._latch(_MESSAGE)

    assert blocked.is_blocked("req-1")
    assert events == ["latched"]


def test_latch_without_a_request_key_warns_and_marks_nothing(
    blocked: BlockedRequests, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(plugin, "_request_key", lambda: "")

    with caplog.at_level(logging.WARNING, logger="agent_hardener.relay_plugin"):
        plugin._latch(_MESSAGE)

    assert "not latching" in caplog.text
    assert not blocked.is_blocked("")


def test_request_key_is_the_propagation_root_uuid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plugin.nemo_relay, "capture_propagation_context", lambda: SimpleNamespace(root_uuid="abc-123"))

    assert _REAL_REQUEST_KEY() == "abc-123"


@pytest.mark.parametrize("root_uuid", [None, ""])
def test_request_key_is_empty_when_there_is_nothing_to_key_on(
    monkeypatch: pytest.MonkeyPatch, root_uuid: str | None
) -> None:
    monkeypatch.setattr(plugin.nemo_relay, "capture_propagation_context", lambda: SimpleNamespace(root_uuid=root_uuid))

    assert _REAL_REQUEST_KEY() == ""


def test_request_key_swallows_a_failing_relay(monkeypatch: pytest.MonkeyPatch) -> None:

    def boom() -> None:
        raise RuntimeError("no active scope")

    monkeypatch.setattr(plugin.nemo_relay, "capture_propagation_context", boom)

    assert _REAL_REQUEST_KEY() == ""


def test_latch_event_recording_never_raises_into_the_tool_path(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("no scope")

    monkeypatch.setattr(plugin.nemo_relay.scope, "event", boom)

    plugin._record_latch()


def test_judge_error_event_carries_the_failing_call(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(plugin.nemo_relay.scope, "event", lambda name, data=None: events.append((name, data)))

    plugin._record_judge_error("rail", "tool", "call-3", "ValueError: nope")

    assert events == [
        (
            "agent_hardener.judge_error",
            {"guardrail": "rail", "tool_name": "tool", "tool_call_id": "call-3", "error": "ValueError: nope"},
        )
    ]


# --- latched tool / llm / stream ---------------------------------------------------------------------------


def test_latched_tool_passes_calls_through_until_the_request_is_blocked(blocked: BlockedRequests) -> None:
    next_call, seen = _downstream()
    intercept = plugin._latched_tool(_MESSAGE)

    allowed = asyncio.run(intercept(_tool_context(), next_call))
    blocked.mark("req-1")
    refused = asyncio.run(intercept(_tool_context(tool_name="send_email", call_id="call-2"), next_call))

    assert allowed.result == {"ok": True}
    assert len(seen) == 1
    assert "earlier guardrail already blocked this request" in str(refused.result)
    assert "send_email" in str(refused.result)


def test_latched_llm_lets_an_unblocked_request_reach_the_provider() -> None:
    next_call, seen = _downstream("provider answer")

    result = asyncio.run(plugin._latched_llm(_MESSAGE)("model", _OPENAI_REQUEST, next_call))

    assert result == "provider answer"
    assert seen == [(_OPENAI_REQUEST,)]


def test_latched_llm_answers_a_blocked_request_locally_exactly_once(blocked: BlockedRequests) -> None:
    blocked.mark("req-1")
    next_call, seen = _downstream("provider answer")
    intercept = plugin._latched_llm(_MESSAGE)

    first = asyncio.run(intercept("model", _OPENAI_REQUEST, next_call))
    second = asyncio.run(intercept("model", _OPENAI_REQUEST, next_call))

    assert first["choices"][0]["message"]["content"] == _MESSAGE
    assert first["choices"][0]["message"]["tool_calls"] == []
    assert second == "provider answer"  # the entry was discarded once the refusal went out
    assert len(seen) == 1
    assert not blocked.is_blocked("req-1")


def test_latched_llm_answers_a_langchain_harness_in_its_own_envelope(blocked: BlockedRequests) -> None:
    blocked.mark("req-1")
    next_call, _ = _downstream()

    payload = asyncio.run(plugin._latched_llm(_MESSAGE)("model", _LANGCHAIN_REQUEST, next_call))

    (envelope,) = payload.values()
    assert envelope["messages"][0]["data"]["content"] == _MESSAGE


def test_latched_llm_fails_open_on_an_unrecognised_request_and_keeps_the_latch(
    blocked: BlockedRequests, caplog: pytest.LogCaptureFixture
) -> None:
    blocked.mark("req-1")
    next_call, seen = _downstream("provider answer")

    with caplog.at_level(logging.WARNING, logger="agent_hardener.relay_plugin"):
        result = asyncio.run(plugin._latched_llm(_MESSAGE)("model", _UNKNOWN_REQUEST, next_call))

    assert result == "provider answer"
    assert len(seen) == 1
    assert "did not recognise the harness response shape" in caplog.text
    assert blocked.is_blocked("req-1")


def _stream_next(chunks: list[Any]) -> Any:
    async def next_call(request: Any) -> AsyncIterator[Any]:
        async def gen() -> AsyncIterator[Any]:
            for chunk in chunks:
                yield chunk

        return gen()

    return next_call


def test_latched_stream_passes_the_providers_chunks_through_when_not_blocked() -> None:
    intercept = plugin._latched_llm_stream(_MESSAGE)

    chunks = asyncio.run(_collect(intercept(_OPENAI_REQUEST, _stream_next(["a", "b"]))))

    assert chunks == ["a", "b"]


def test_latched_stream_replays_the_refusal_when_blocked(blocked: BlockedRequests) -> None:
    blocked.mark("req-1")
    intercept = plugin._latched_llm_stream(_MESSAGE)

    chunks = asyncio.run(_collect(intercept(_OPENAI_REQUEST, _stream_next(["provider"]))))

    assert [chunk["choices"][0]["finish_reason"] for chunk in chunks] == [None, "stop"]
    assert chunks[0]["choices"][0]["delta"]["content"] == _MESSAGE
    assert not blocked.is_blocked("req-1")


@pytest.mark.parametrize("request_", [_UNKNOWN_REQUEST, _LANGCHAIN_REQUEST], ids=["unknown", "langchain"])
def test_latched_stream_fails_open_without_an_openai_shaped_request(
    blocked: BlockedRequests, caplog: pytest.LogCaptureFixture, request_: Any
) -> None:
    blocked.mark("req-1")
    intercept = plugin._latched_llm_stream(_MESSAGE)

    with caplog.at_level(logging.WARNING, logger="agent_hardener.relay_plugin"):
        chunks = asyncio.run(_collect(intercept(request_, _stream_next(["provider"]))))

    assert chunks == ["provider"]
    assert "did not recognise the harness stream shape" in caplog.text
    assert blocked.is_blocked("req-1")


# --- outcome construction ----------------------------------------------------------------------------------


@pytest.mark.parametrize("takes_annotation", [True, False])
def test_forward_unwraps_a_relay_result_and_keeps_the_annotation_when_supported(
    monkeypatch: pytest.MonkeyPatch, takes_annotation: bool
) -> None:
    built: list[tuple[Any, dict[str, Any]]] = []

    def outcome(payload: Any, **kwargs: Any) -> str:
        built.append((payload, kwargs))
        return "outcome"

    monkeypatch.setattr(plugin.nemo_relay, "ToolExecutionInterceptOutcome", outcome)
    monkeypatch.setattr(plugin, "_OUTCOME_TAKES_ANNOTATION", takes_annotation)

    assert plugin._forward(SimpleNamespace(result={"v": 1}, annotation="note")) == "outcome"

    assert built == [({"v": 1}, {"annotation": "note"} if takes_annotation else {})]


def test_forward_accepts_a_bare_payload_from_harnesses_without_a_result_attribute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[Any, dict[str, Any]]] = []
    monkeypatch.setattr(
        plugin.nemo_relay, "ToolExecutionInterceptOutcome", lambda payload, **kw: built.append((payload, kw))
    )
    monkeypatch.setattr(plugin, "_OUTCOME_TAKES_ANNOTATION", True)

    plugin._forward({"plain": "dict"})

    assert built == [({"plain": "dict"}, {"annotation": None})]


def test_refusal_payload_is_a_tool_message_answering_the_recorded_call() -> None:
    payload = plugin._refusal_payload("blocked", "transfer_funds", "call-7")

    message = BestEffortAnyCodec().from_json(payload)
    assert message.tool_call_id == "call-7"
    assert message.content == "blocked"


def test_refusal_payload_falls_back_to_the_tool_name_without_a_call_id() -> None:
    message = BestEffortAnyCodec().from_json(plugin._refusal_payload("blocked", "transfer_funds", None))

    assert message.tool_call_id == "transfer_funds"


def test_refusal_payload_is_the_plain_reason_when_langchain_is_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langchain_core.messages", None)

    assert plugin._refusal_payload("blocked", "tool", "id") == "blocked"


# --- registration ------------------------------------------------------------------------------------------


class _Context:
    def __init__(self) -> None:
        self.registered: list[tuple[str, str, int]] = []

    def register_tool_execution_intercept(self, name: str, priority: int, _callback: Any) -> None:
        self.registered.append(("tool", name, priority))

    def register_llm_execution_intercept(self, name: str, priority: int, _callback: Any) -> None:
        self.registered.append(("llm", name, priority))

    def register_llm_stream_execution_intercept(self, name: str, priority: int, _callback: Any) -> None:
        self.registered.append(("stream", name, priority))


def _register(config: dict[str, Any], context: _Context) -> None:
    plugin.AgentHardenerGuardrailPlugin().register(config, cast("PluginContext", context))


@pytest.fixture
def fake_judge(monkeypatch: pytest.MonkeyPatch, make_judge: Callable[[float | Exception], Any]) -> None:
    monkeypatch.setattr(plugin, "LlmSafetyJudge", lambda _model: make_judge(0.0))


def test_register_installs_rails_in_authoring_order_ahead_of_the_latch(
    fake_judge: None, plugin_config: Callable[..., dict[str, Any]]
) -> None:
    del fake_judge
    context = _Context()

    _register(plugin_config(count=2), context)

    tools = [(name, priority) for kind, name, priority in context.registered if kind == "tool"]
    rails = [(name, priority) for name, priority in tools if name != plugin._LATCH_NAME]
    (latch_priority,) = (priority for name, priority in tools if name == plugin._LATCH_NAME)
    assert [name for name, _ in rails] == ["custom_guardrail_1", "custom_guardrail_2"]
    assert [priority for _, priority in rails] == sorted(priority for _, priority in rails)
    assert all(priority > latch_priority for _, priority in rails)
    assert {kind for kind, name, _ in context.registered if name == plugin._LATCH_NAME} == {"tool", "llm", "stream"}


def test_register_leaves_the_latch_out_when_disabled(
    fake_judge: None, plugin_config: Callable[..., dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_judge
    monkeypatch.setattr(plugin, "_LATCH_DISABLED", True)
    context = _Context()

    _register(plugin_config(), context)

    assert [(kind, name) for kind, name, _ in context.registered] == [("tool", "custom_guardrail_1")]


def test_register_refuses_to_start_on_a_relay_without_tool_execution_context(
    fake_judge: None, plugin_config: Callable[..., dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    del fake_judge
    monkeypatch.delattr(plugin.nemo_relay, "ToolExecutionContext")
    context = _Context()

    with pytest.raises(RuntimeError, match=r"nemo-relay 0\.9 or newer"):
        _register(plugin_config(), context)

    assert context.registered == []


def test_register_function_registers_the_plugin_kind_with_relay(monkeypatch: pytest.MonkeyPatch) -> None:
    registered: list[tuple[str, Any]] = []
    monkeypatch.setattr(plugin.nemo_relay.plugin, "register", lambda kind, impl: registered.append((kind, impl)))

    plugin.register()

    ((kind, impl),) = registered
    assert kind == plugin.PLUGIN_KIND
    assert isinstance(impl, plugin.AgentHardenerGuardrailPlugin)
