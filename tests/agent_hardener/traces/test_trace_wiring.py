# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provider reaches exactly the agents that consume traces, and nothing changes without one."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    AgentRunOutput,
    AttackRecord,
    RunSettings,
    SessionConfig,
    StorageConfig,
    TargetInput,
    TracesConfig,
    ValidatorReport,
    VictimResult,
)
from agent_hardener.runtime.orchestrator import Orchestrator
from agent_hardener.runtime.runner import Mission
from agent_hardener.traces.relay import RelayTraceProvider
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.runtime.run_context import RunContext


class CapturingAdapter:
    """Answers like a happy run and records the trace provider each agent was handed."""

    def __init__(self) -> None:
        self.providers: dict[str, object] = {}

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: RunContext | None = None) -> AgentRunOutput:
        self.providers[agent.name] = ctx.trace_provider if ctx else "no ctx"
        if agent.role == "attacker":
            return AttackRecord(agent_id=agent.agent_id, agent_name=agent.name, records=[{"prompt": "p"}])
        if agent.role == "victim":
            return VictimResult(agent_id=agent.agent_id, agent_name=agent.name, ok=True)
        return ValidatorReport(agent_id=agent.agent_id, agent_name=agent.name, kind=agent.config["kind"], ok=True)


def _config(tmp_path: Path) -> SessionConfig:
    return SessionConfig(
        storage=StorageConfig(root_dir=tmp_path),
        run=RunSettings(rounds=1),
        target=TargetInput(name="target", base_url="http://target.local"),
        attackers=[AgentConfig(name="attacker", role="attacker")],
        victim=AgentConfig(name="victim", role="victim"),
        attack_validators=[AgentConfig(name="attack", role="validator", config={"kind": "attack"})],
        benign_validators=[AgentConfig(name="benign", role="validator", config={"kind": "benign"})],
    )


def _run(tmp_path: Path, provider: FakeTraceProvider | None) -> tuple[CapturingAdapter, bool]:
    adapter = CapturingAdapter()
    orchestrator = Orchestrator(_config(tmp_path), agent_adapter=adapter, trace_provider=provider)
    report = asyncio.run(orchestrator.run_round())
    return adapter, report.success


def test_without_a_provider_no_agent_gets_one(tmp_path: Path) -> None:
    adapter, success = _run(tmp_path, None)
    assert success
    assert set(adapter.providers.values()) == {None}


def test_every_agent_gets_the_provider(tmp_path: Path) -> None:
    provider = FakeTraceProvider()
    adapter, success = _run(tmp_path, provider)
    assert success
    assert adapter.providers == {"attacker": provider, "victim": provider, "attack": provider, "benign": provider}


def _mission(tmp_path: Path, relay_victim: object) -> Mission:
    mission = Mission.__new__(Mission)
    mission.ctx = SimpleNamespace(
        session_config=SimpleNamespace(traces=TracesConfig(), storage=SimpleNamespace(root_dir=tmp_path))
    )
    mission._run_id = "run-1"
    mission._base_config = SimpleNamespace(relay_victim=relay_victim, cwd=tmp_path)
    mission._announce = lambda _message: None
    return mission


def test_runner_builds_the_relay_provider_for_a_relay_victim(tmp_path: Path) -> None:
    spec = SimpleNamespace(atof_path=tmp_path / "artifacts" / "events.atof.jsonl")
    provider = _mission(tmp_path, spec)._build_trace_provider()
    assert isinstance(provider, RelayTraceProvider)
    assert (tmp_path / "run-logs" / "run-1" / "traces").is_dir()


def test_runner_builds_no_provider_for_a_plain_http_target(tmp_path: Path) -> None:
    assert _mission(tmp_path, None)._build_trace_provider() is None
