# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Recognise what our own defenses leave behind in a tool call or a model answer.

These patterns describe our guardrail plugin and the OpenShell egress proxy, not any trace backend,
so every provider classifies with them instead of carrying its own copy.

Two of them are inferred from code rather than seen in a real run and should be pinned against a
captured stream: the exact body of the proxy's 403, and how the latch mark is recorded.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_hardener.models.traces import ToolOutcome

#: ``agent-hardener[<rail>]: refused <tool> (...)`` (relay_plugin/policy.py) names the rail that fired;
#: ``agent-hardener: refused <tool> (an earlier guardrail ...)`` (relay_plugin/plugin.py) is the latch,
#: which refuses every later tool of a request a rail already blocked, and names no rail.
_RAIL_REFUSAL = re.compile(r"agent-hardener\[(?P<rail>[^\]]+)\]: refused\b")
_LATCH_REFUSAL = re.compile(r"agent-hardener: refused\b")

#: The canned answer a blocked request gets instead of a model call (relay_plugin/blocked.py). The
#: message is copied rather than imported: importing ``relay_plugin`` loads Relay itself. A test keeps
#: it equal to ``relay_plugin.config.DEFAULT_BLOCKED_MESSAGE``.
DEFAULT_BLOCKED_MESSAGE = "The request was blocked by security policy."
BLOCKED_RESPONSE_ID = "agent-hardener-blocked"
BLOCKED_RESPONSE_MARKERS = (DEFAULT_BLOCKED_MESSAGE, BLOCKED_RESPONSE_ID)

#: The Relay mark the plugin records when it latches a request (relay_plugin/plugin.py).
LATCH_EVENT = "agent_hardener.latched"

#: How an egress denial by the OpenShell proxy reaches a tool: its 403, or a refused/failed proxy
#: connection. OpenShell writes no trace event for a denial, so this text is the only evidence.
NETWORK_DENIAL_PATTERNS = (
    re.compile(r"\b403\b[^\n]{0,80}\bforbidden\b", re.IGNORECASE),
    re.compile(r"\bforbidden\b[^\n]{0,80}\b403\b", re.IGNORECASE),
    re.compile(r"\bconnection refused\b", re.IGNORECASE),
    re.compile(r"\bproxy ?error\b", re.IGNORECASE),
    re.compile(r"\btunnel connection failed\b", re.IGNORECASE),
)


def text_of(value: Any) -> str:
    """Flatten a recorded payload to text so the patterns can search it, whatever its shape."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def rail_of(result: Any) -> str | None:
    """The rail named in a guardrail refusal, or ``None`` when there is none (or it was the latch)."""
    match = _RAIL_REFUSAL.search(text_of(result))
    return match.group("rail") if match else None


def is_guardrail_refusal(result: Any) -> bool:
    """Whether the tool result is a refusal written by our guardrail plugin."""
    text = text_of(result)
    return bool(_RAIL_REFUSAL.search(text) or _LATCH_REFUSAL.search(text))


def is_network_denial(result: Any) -> bool:
    """Whether the tool result carries the egress proxy's denial."""
    text = text_of(result)
    return any(pattern.search(text) for pattern in NETWORK_DENIAL_PATTERNS)


def is_blocked_response(output: Any, blocked_message: str | None = None) -> bool:
    """Whether a model answer is the canned blocked response rather than a real model call."""
    text = text_of(output)
    markers = (*BLOCKED_RESPONSE_MARKERS, blocked_message) if blocked_message else BLOCKED_RESPONSE_MARKERS
    return any(marker in text for marker in markers)


def classify_tool_call(result: Any, succeeded: bool) -> tuple[ToolOutcome, str | None]:
    """The outcome of a tool call and the rail that refused it, if any.

    ``succeeded`` is the backend's own error status. It is not enough on its own: our guardrail
    returns its refusal as the tool result instead of raising, so a refused call ends successfully
    and only the result text shows it. Checked in order: guardrail, then network, then the status.
    """
    if is_guardrail_refusal(result):
        return "guardrail_refused", rail_of(result)
    if is_network_denial(result):
        return "network_denied", None
    return ("ok" if succeeded else "error"), None
