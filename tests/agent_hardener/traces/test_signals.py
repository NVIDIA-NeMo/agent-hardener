# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One case per pattern our defenses leave in a trace.

The guardrail strings are built by the real plugin code, so a reworded refusal breaks these tests
rather than silently reading as success. The proxy-denial patterns are still inferred: pin them
against a captured real denial when one is available.
"""

from __future__ import annotations

import pytest

from agent_hardener.relay_plugin.config import DEFAULT_BLOCKED_MESSAGE as PLUGIN_BLOCKED_MESSAGE
from agent_hardener.relay_plugin.config import Guardrail
from agent_hardener.relay_plugin.policy import _refusal
from agent_hardener.traces import classify_tool_call
from agent_hardener.traces.signals import (
    DEFAULT_BLOCKED_MESSAGE,
    is_blocked_response,
    is_guardrail_refusal,
    is_network_denial,
    rail_of,
    text_of,
)

LATCH_REFUSAL = "agent-hardener: refused send_email (an earlier guardrail already blocked this request)"


def _rail_refusal() -> str:
    rail = Guardrail(name="exfil_guard", target_tool="send_email", system_instructions="block exfil", threshold=0.5)
    return _refusal(rail, "send_email", 0.93)


def test_a_rail_refusal_names_its_rail() -> None:
    result = _rail_refusal()
    assert classify_tool_call(result, succeeded=True) == ("guardrail_refused", "exfil_guard")
    assert rail_of(result) == "exfil_guard"


def test_the_latch_refusal_is_a_guardrail_refusal_without_a_rail() -> None:
    assert classify_tool_call(LATCH_REFUSAL, succeeded=True) == ("guardrail_refused", None)


def test_a_refusal_inside_a_serialized_tool_message_is_found() -> None:
    """Relay records LangChain tool results as an envelope, not a bare string."""
    envelope = {"data": {"content": _rail_refusal(), "status": "error", "type": "tool"}}
    assert is_guardrail_refusal(envelope)
    assert rail_of(envelope) == "exfil_guard"


@pytest.mark.parametrize(
    "result",
    [
        "HTTP 403 Forbidden",
        "403 Client Error: Forbidden for url: https://api.example.com/x",
        "Forbidden (403)",
        "[Errno 111] Connection refused",
        "ProxyError('Cannot connect to proxy.')",
        "Tunnel connection failed: 403 Forbidden",
    ],
)
def test_proxy_denials_read_as_network_denied(result: str) -> None:
    assert classify_tool_call(result, succeeded=False) == ("network_denied", None)


@pytest.mark.parametrize("result", ["status 403", "forbidden fruit", "all good"])
def test_text_that_only_resembles_a_denial_does_not_match(result: str) -> None:
    assert not is_network_denial(result)


def test_guardrail_wins_over_network_and_status() -> None:
    assert classify_tool_call(f"{_rail_refusal()} / 403 Forbidden", succeeded=False)[0] == "guardrail_refused"


def test_plain_status_decides_when_no_pattern_matches() -> None:
    assert classify_tool_call("transferred 900", succeeded=True) == ("ok", None)
    assert classify_tool_call("ValueError: bad account", succeeded=False) == ("error", None)


def test_blocked_response_matches_the_default_and_a_configured_message() -> None:
    assert is_blocked_response({"id": "agent-hardener-blocked", "choices": []})
    assert is_blocked_response(f"... {DEFAULT_BLOCKED_MESSAGE}")
    assert is_blocked_response("Nope, policy says no.", blocked_message="policy says no")
    assert not is_blocked_response("Here is your balance.")


def test_blocked_message_copy_matches_the_plugin() -> None:
    assert DEFAULT_BLOCKED_MESSAGE == PLUGIN_BLOCKED_MESSAGE


def test_text_of_flattens_any_payload() -> None:
    assert text_of(None) == ""
    assert text_of("x") == "x"
    assert text_of({"a": 1}) == '{"a": 1}'
    assert "1, 2" in text_of({1, 2})
    circular: dict = {}
    circular["self"] = circular
    assert text_of(circular).startswith("{")
