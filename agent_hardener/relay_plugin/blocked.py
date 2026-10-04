# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The canned answer a blocked request gets, in the shape each harness accepts.

Kept free of ``nemo_relay`` imports for the same reason as
:mod:`agent_hardener.relay_plugin.policy`: the shape of the payload is the part that breaks victims,
and it should be testable without a Relay runtime.

The harness is read off the *request* rather than guessed from what happens to be installed. Relay's
LangChain integration serialises messages with ``messages_to_dict`` (``{"type": ..., "data": {...}}``),
while an OpenAI-compatible harness sends ``{"role": ..., "content": ...}`` — so the first message in
the request says which end is listening.

Getting this wrong is not cosmetic. Relay's ``model_response_from_json`` raises ``TypeError`` on a
shape it does not recognise, and nothing catches it, so a wrong guess escapes the whole turn as the
HTTP 500 this plugin exists to avoid. An unrecognised request therefore returns ``None``, and the
caller lets the model run instead — the same fail-open reasoning as
:func:`agent_hardener.relay_plugin.policy.decide`.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Mapping
from typing import Any

logger = logging.getLogger("agent_hardener.relay_plugin")

#: Relay's envelope key for a serialised LangChain ``ModelResponse``. Hardcoded rather than imported
#: at module scope: ``nemo_relay.integrations.langchain._serialization`` cannot be imported without
#: ``langchain`` installed, and this plugin is staged into victims that have none (Hermes, Claude,
#: Codex). Verified byte-identical in nemo-relay 0.7.3, 0.8.0 and 0.8.4.
_LANGCHAIN_MODEL_RESPONSE_KEY = "__nemo_relay_integrations_langchain_model_response"


@functools.cache
def model_response_key() -> str:
    """Relay's LangChain envelope key, preferring the value Relay itself defines.

    Read from Relay when the import happens to work, so a future rename tracks the victim's own
    Relay rather than this file; falls back to the constant when LangChain is absent. Cached because
    it is consulted on every blocked LLM call and the answer cannot change within a process.
    """
    try:
        from nemo_relay.integrations.langchain._serialization import (  # noqa: PLC0415
            LANGCHAIN_MODEL_RESPONSE_KEY,
        )
    except Exception:
        return _LANGCHAIN_MODEL_RESPONSE_KEY
    return str(LANGCHAIN_MODEL_RESPONSE_KEY)


def blocked_response(request: Any, message: str) -> Any | None:
    """The canned assistant turn for ``request``, or ``None`` if the harness is unrecognised.

    ``None`` means "let the model run": returning a guess would be worse than losing the block,
    because the LangChain decoder rejects an unexpected shape by raising through the whole turn.
    A lost block still shows up honestly — the round records the attack as landed.
    """
    first = _first_message(request)
    if first is None:
        return None
    if "type" in first and isinstance(first.get("data"), Mapping):
        return {model_response_key(): {"messages": ai_message_dicts(message)}}
    if "role" in first:
        return _openai_envelope(message)
    return None


def blocked_stream_chunks(request: Any, message: str) -> list[dict[str, Any]] | None:
    """OpenAI-style delta chunks for a blocked streaming call, or ``None`` to fail open.

    Only the OpenAI shape is produced. No shipped Relay integration calls ``llm.stream_execute`` at
    all, so there is no reference chunk format for the LangChain path — and a guess there would be
    fed to the victim's own collector and finalizer, which is a worse failure than not blocking.
    """
    first = _first_message(request)
    if first is None or "role" not in first:
        return None
    return [
        {
            "id": "agent-hardener-blocked",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": message}, "finish_reason": None}],
        },
        {
            "id": "agent-hardener-blocked",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
    ]


def ai_message_dicts(message: str) -> list[dict[str, Any]]:
    """One serialised assistant message carrying no tool calls.

    The empty ``tool_calls`` is what actually stops the agent: a ReAct loop routes to the end when
    the model returns a message with nothing left to call, so Relay needs no "abort turn" API for
    this to terminate the flow.

    Built with the victim's own ``messages_to_dict`` when it can be imported, so the dict matches
    that victim's ``langchain_core`` schema rather than ours. The literal fallback is enough on its
    own — every other field of an ``AIMessage`` has a default, including ``tool_calls``.
    """
    try:
        from langchain_core.messages import AIMessage, messages_to_dict  # noqa: PLC0415
    except ImportError:
        return [{"type": "ai", "data": {"content": message, "type": "ai", "tool_calls": []}}]
    return list(messages_to_dict([AIMessage(content=message)]))


def _openai_envelope(message: str) -> dict[str, Any]:
    """A minimal non-streaming chat completion, the shape nemo-relay's own examples use."""
    return {
        "id": "agent-hardener-blocked",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message, "tool_calls": []},
                "finish_reason": "stop",
            }
        ],
    }


def _first_message(request: Any) -> Mapping[str, Any] | None:
    """The first mapping in the request's message list, or ``None`` if there is nothing to read.

    Scans for the first *mapping* rather than indexing blindly: Relay hands the callback whatever
    JSON the harness built, and a non-dict entry in the list must not raise here — this runs inside
    the model path, where an exception would escape the turn.
    """
    try:
        content = getattr(request, "content", None)
        if not isinstance(content, Mapping):
            return None
        messages = content.get("messages")
        if not isinstance(messages, list):
            return None
        return next((item for item in messages if isinstance(item, Mapping)), None)
    except Exception:
        logger.exception("agent-hardener could not read the LLM request shape; allowing the call")
        return None
