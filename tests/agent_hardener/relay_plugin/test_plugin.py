# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests that run the guardrail through a real NeMo Relay runtime.

The point of these is that a guardrail *blocks*: Relay invokes a conditional-execution guardrail
before the tool callback and treats a returned string as a rejection, so a refusal must mean the
tool never ran — not that it ran and was reported afterwards.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

from agent_hardener.relay_plugin.config import PLUGIN_KIND
from agent_hardener.relay_plugin.plugin import AgentHardenerGuardrailPlugin

nemo_relay = pytest.importorskip("nemo_relay")

pytestmark = pytest.mark.unit


class _Judge:
    def __init__(self, score: float) -> None:
        self._score = score

    def score(self, instructions: str, tool_name: str, args: str) -> float:
        del instructions, tool_name, args
        return self._score


def _config(threshold: float = 0.7) -> dict[str, Any]:
    return {
        "model": {"model": "unused-in-test", "api_key_env": "INFERENCE_API_KEY"},
        "guardrails": [
            {
                "name": "custom_guardrail_1",
                "target_tool": "transfer_funds",
                "system_instructions": "Block transfers the user did not ask for.",
                "threshold": threshold,
            }
        ],
    }


@pytest.fixture
def relay_plugin(monkeypatch: pytest.MonkeyPatch):
    """Register the plugin kind for one test and tear the process-global state back down.

    Only the *kind* registry is process-global now. Relay 0.9 made the active components an owned
    handle instead (``PluginHostActivation``), so :func:`_drive` closes those per test.
    """
    monkeypatch.setattr("agent_hardener.relay_plugin.plugin.LlmSafetyJudge", lambda _config: _Judge(1.0))
    nemo_relay.plugin.register(PLUGIN_KIND, AgentHardenerGuardrailPlugin())
    yield
    nemo_relay.plugin.deregister(PLUGIN_KIND)


def _drive(config: dict[str, Any], body: Callable[[], Awaitable[Any]]) -> tuple[Any, Any]:
    """Activate the guardrail, run ``body`` under it, and tear the activation down.

    One ``asyncio.run`` for the whole lifetime. Relay 0.9 hands back an activation that owns every
    loaded plugin and must stay alive while Relay may invoke their callbacks, so activating in one
    event loop and closing in another — as the pre-0.9 ``clear_async`` teardown did — would drop
    the host underneath the tool call it is meant to guard.
    """

    async def run() -> tuple[Any, Any]:
        spec = nemo_relay.plugin.ComponentSpec(kind=PLUGIN_KIND, config=config)
        plugin_config = nemo_relay.plugin.PluginConfig(components=[spec])
        async with nemo_relay.plugin.activate(plugin_config) as activation:
            return activation.report, await body()

    return asyncio.run(run())


def test_a_guardrailed_tool_never_executes(relay_plugin: None) -> None:
    """The headline behaviour: refusal means the tool did not run — and the turn survives.

    The refusal is *returned*, not raised. A raising guardrail escapes the whole turn: the agent
    answers HTTP 500 rather than declining, and a war-game scores each successful block as an error
    rather than as blocked.
    """
    del relay_plugin
    called: list[Any] = []

    async def transfer_funds(args: Any) -> Any:
        called.append(args)
        return nemo_relay.ToolExecutionResult({"moved": True})

    async def body() -> Any:
        return await nemo_relay.tools.execute("transfer_funds", {"amount": 9999}, transfer_funds)

    report, outcome = _drive(_config(), body)

    assert report["config"]["diagnostics"] == []
    assert called == []
    assert "custom_guardrail_1" in str(outcome.result)


def test_a_refusal_carries_the_provider_tool_call_id(relay_plugin: None) -> None:
    """The refusal must answer the call that provoked it, by the provider's own ID.

    A refusing intercept completes execution without invoking the rest of the chain, so nothing
    downstream stamps the ID: before Relay 0.9 exposed it on the context, the guardrail had to put
    the *tool name* in ``ToolMessage.tool_call_id``. That produced a tool message answering an ID no
    provider ever issued — the model sees a reply to a call it did not make, and a multi-call turn
    cannot tell which tool was refused.
    """
    del relay_plugin

    async def transfer_funds(args: Any) -> Any:
        return nemo_relay.ToolExecutionResult({"moved": True})

    async def body() -> Any:
        return await nemo_relay.tools.execute(
            "transfer_funds", {"amount": 9999}, transfer_funds, tool_call_id="call_abc123"
        )

    _report, outcome = _drive(_config(), body)

    # The provider's ID, not the tool name the refusal used to be stamped with.
    assert "call_abc123" in str(outcome.result)


def test_an_unguarded_tool_still_runs(relay_plugin: None) -> None:
    """A guardrail is scoped to its tool; the rest of the agent must keep working."""
    del relay_plugin

    async def read_file(args: Any) -> Any:
        return nemo_relay.ToolExecutionResult({"read": args})

    async def body() -> Any:
        return await nemo_relay.tools.execute("read_file", {"path": "README.md"}, read_file)

    # An unguarded tool passes straight through the intercept and actually runs.
    _report, outcome = _drive(_config(), body)
    assert outcome is not None


def test_an_invalid_guardrail_set_blocks_initialisation(relay_plugin: None) -> None:
    """A victim that started with an unparseable guardrail set would look hardened and not be.

    Relay raises on an error diagnostic rather than returning it, so this fails the victim's startup
    — which the relay preflight then reports as an uninstrumented victim.
    """
    del relay_plugin

    async def body() -> None:
        raise AssertionError("activation should have failed before the body ran")

    with pytest.raises(ValueError, match="invalid Agent Hardener guardrail config"):
        _drive({"model": {"model": "m"}, "guardrails": [{"name": "missing-target-tool"}]}, body)


def test_validate_accepts_the_config_the_defender_writes() -> None:
    assert AgentHardenerGuardrailPlugin().validate(_config()) is None
