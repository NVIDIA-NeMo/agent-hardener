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
        attackers=[AgentConfig(name="attacker", role="attacker", config={"use_traces": True})],
        victim=AgentConfig(name="victim", role="victim"),
        attack_validators=[
            AgentConfig(name="attack-on", role="validator", config={"kind": "attack", "use_traces": True})
        ],
        benign_validators=[AgentConfig(name="benign-off", role="validator", config={"kind": "benign"})],
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


def test_only_agents_with_use_traces_get_the_provider(tmp_path: Path) -> None:
    provider = FakeTraceProvider()
    adapter, success = _run(tmp_path, provider)
    assert success
    assert adapter.providers == {"attacker": provider, "victim": None, "attack-on": provider, "benign-off": None}


def _mission(tmp_path: Path, provider: str, relay_victim: object) -> Mission:
    mission = Mission.__new__(Mission)
    mission.ctx = SimpleNamespace(
        session_config=SimpleNamespace(
            traces=TracesConfig(provider=provider), storage=SimpleNamespace(root_dir=tmp_path)
        )
    )
    mission._run_id = "run-1"
    mission._base_config = SimpleNamespace(relay_victim=relay_victim, cwd=tmp_path)
    mission._announce = lambda _message: None
    return mission


def test_runner_builds_no_provider_and_no_traces_dir_when_off(tmp_path: Path) -> None:
    assert _mission(tmp_path, "none", object())._build_trace_provider() is None
    assert not list(tmp_path.rglob("traces"))


def test_runner_builds_the_relay_provider_for_a_relay_victim(tmp_path: Path) -> None:
    spec = SimpleNamespace(atof_path=tmp_path / "artifacts" / "events.atof.jsonl")
    provider = _mission(tmp_path, "relay", spec)._build_trace_provider()
    assert isinstance(provider, RelayTraceProvider)
    assert (tmp_path / "run-logs" / "run-1" / "traces").is_dir()
    assert _mission(tmp_path, "relay", None)._build_trace_provider() is None
