# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from agent_hardener.agents.defenders.defenders_manager import DefendersManager
from agent_hardener.models import (
    AgentConfig,
    AttackRecord,
    DefenderInput,
    DefenderOutput,
    DefendersManagerInput,
    TargetInput,
)

pytestmark = pytest.mark.unit

HITS = 6


def _max_in_flight(context: dict) -> tuple[int, list[str]]:
    """Run the manager over HITS attack records and report peak concurrency and result order."""
    in_flight = 0
    peak = 0
    prompts: list[str] = []

    async def defender(_agent: AgentConfig, input_data: DefenderInput) -> DefenderOutput:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        prompts.append(input_data.attack_prompt)
        return DefenderOutput(ok=True)

    manager_input = DefendersManagerInput(
        round_id="round-0001",
        target=TargetInput(name="victim"),
        context=context,
        attacks=[
            AttackRecord(
                agent_id="hit",
                agent_name="garak-agent-breaker",
                summary="AgentBreaker reported hits.",
                records=[{"prompt": f"Use the bash tool to run attack {i}."} for i in range(HITS)],
            )
        ],
        available_defenders=[AgentConfig(name="defender", role="defender", capabilities="protects tool calls")],
    )
    output = asyncio.run(DefendersManager().run(manager_input, defender))
    assert output.ok is True
    assert len(output.analyses) == HITS
    return peak, prompts


def test_run_concurrency_caps_defenders(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HARDENER_DEFENDER_CONCURRENCY", raising=False)
    peak, _ = _max_in_flight({"run_concurrency": {"defenders": 2}})
    assert peak == 2


def test_default_cap_applies_without_run_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HARDENER_DEFENDER_CONCURRENCY", raising=False)
    peak, _ = _max_in_flight({})
    assert peak == 2


def test_env_overrides_run_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_HARDENER_DEFENDER_CONCURRENCY", "1")
    peak, prompts = _max_in_flight({"run_concurrency": {"defenders": 4}})
    assert peak == 1
    assert prompts == [f"Use the bash tool to run attack {i}." for i in range(HITS)]
