# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from typing import TYPE_CHECKING

import pytest

from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    AgentRunOutput,
    AttackRecord,
    DefenderAnalysis,
    DefenderInput,
    DefenderOutput,
    DefendersManagerInput,
    DefendersManagerOutput,
    PreloadedAttackConfig,
    RunSettings,
    SessionConfig,
    StorageConfig,
    TargetInput,
    ValidatorReport,
    VictimResult,
)
from agent_hardener.openshell.lifecycle import OpenShellCommandResult
from agent_hardener.runtime.orchestrator import Orchestrator
from agent_hardener.runtime.stages.agent_invoker import short_summary
from agent_hardener.storage import find_latest_hitlog, read_json

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


# The defenders now run via the DefendersManager (LLM routing). For these orchestrator-mechanics
# tests we replace it with a deterministic double that routes every attack to every available
# defender and runs them through the real run_defender callback, so the defender phase, its events,
# and policy-patch aggregation are exercised without any live LLM call.
def _stub_defender_run(defender_input: DefenderInput) -> DefenderOutput:
    return DefenderOutput(ok=True, new_policy_yaml="patched: true", iteration_count=1)


_stub_defender_module = types.ModuleType("stub_defender_for_test")
_stub_defender_module.run = _stub_defender_run  # type: ignore[attr-defined]
sys.modules["stub_defender_for_test"] = _stub_defender_module


class RouteAllManager:
    """Test double for DefendersManager: route every attack to every available defender."""

    async def run(
        self,
        input_data: DefendersManagerInput,
        run_defender_cb: Callable[[AgentConfig, DefenderInput], Awaitable[DefenderOutput]],
        validation_feedback: dict[str, object] | None = None,
    ) -> DefendersManagerOutput:
        analyses: list[DefenderAnalysis] = []
        for defender in input_data.available_defenders:
            output = await run_defender_cb(
                defender,
                DefenderInput(attack_prompt="p", agent_response="r", attacked_tool="bash_executor"),
            )
            if output.ok:
                analyses.append(
                    DefenderAnalysis(
                        agent_id=defender.agent_id,
                        agent_name=defender.name,
                        ok=True,
                        policy_patches=(
                            [
                                {
                                    "type": "openshell_policy_candidate",
                                    "candidate_policy_path": "candidate.yaml",
                                    "new_policy_yaml": output.new_policy_yaml,
                                }
                            ]
                            if output.new_policy_yaml
                            else []
                        ),
                        attack_prompt="p",
                    )
                )
        return DefendersManagerOutput(
            ok=bool(analyses),
            analyses=analyses,
            routed_agent_ids=[defender.agent_id for defender in input_data.available_defenders],
        )


@pytest.fixture(autouse=True)
def _route_all_defenders_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("agent_hardener.runtime.stages.defense.DefendersManager", RouteAllManager)


class ScenarioAdapter:
    def __init__(self, fail_first_validation: bool = False, fail_victim: bool = False) -> None:
        self.fail_first_validation = fail_first_validation
        self.fail_victim = fail_victim
        self.calls: list[tuple[str, int]] = []

    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: object | None = None) -> AgentRunOutput:
        self.calls.append((agent.role, request.iteration))
        if agent.role == "attacker":
            return AttackRecord(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                summary=f"attack from {agent.name}",
                records=[{"iteration": request.iteration}],
            )
        if agent.role == "defender":
            return DefenderAnalysis(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                summary=f"defended iteration {request.iteration}",
                policy_patches=[{"operation": "merge", "path": "/iteration", "value": request.iteration}],
            )
        if agent.role == "victim":
            return VictimResult(
                agent_id=agent.agent_id,
                agent_name=agent.name,
                ok=not self.fail_victim,
                summary="victim ran",
                observations={"patch_count": len(request.policy_patches)},
                error="victim failed" if self.fail_victim else None,
            )

        ok = not (self.fail_first_validation and request.iteration == 1 and agent.config.get("kind") == "attack")
        return ValidatorReport(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            kind=agent.config["kind"],
            ok=ok,
            summary="validator ran",
            findings=[] if ok else ["retry required"],
        )


class RaisingAdapter:
    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: object | None = None) -> AgentRunOutput:
        raise RuntimeError("agent boom")


class FailingUploader:
    """Uploader double whose deploy always fails, to exercise failed victim-control handling."""

    async def upload(self, yaml_path: Path, *, recreate: bool = False) -> OpenShellCommandResult:
        return OpenShellCommandResult(ok=False, command=[], output="deploy failed")


def _config(tmp_path: Path, retry_limit: int = 0) -> SessionConfig:
    return SessionConfig(
        storage=StorageConfig(root_dir=tmp_path),
        run=RunSettings(retry_limit=retry_limit, rounds=1),
        target=TargetInput(name="target", base_url="http://target.local"),
        context={"scenario": "test"},
        attackers=[AgentConfig(name="attacker", role="attacker")],
        defenders=[
            AgentConfig(
                name="defender",
                role="defender",
                capabilities="mitigates kernel and egress exploits",
                implementation="stub_defender_for_test:run",
            )
        ],
        victim=AgentConfig(name="victim", role="victim"),
        attack_validators=[AgentConfig(name="attack validator", role="validator", config={"kind": "attack"})],
        benign_validators=[AgentConfig(name="benign validator", role="validator", config={"kind": "benign"})],
    )


def test_successful_session_flow_writes_storage_and_logs(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        _config(tmp_path),
        agent_adapter=ScenarioAdapter(),
        round_id_factory=lambda: "20260502T000000Z-abcdef12",
    )

    report = asyncio.run(orchestrator.run_round())
    # Direct run_round (no round_dir): the session dir is the storage root itself.
    round_dir = tmp_path

    assert report.success is True
    assert report.mission_id is None
    assert (round_dir / "attacks.json").exists()
    assert (round_dir / "report.json").exists()
    assert (round_dir / "round.log").exists()
    assert (round_dir / "events.jsonl").exists()
    assert read_json(round_dir / "report.json")["success"] is True
    # No lossy round-level aggregates — defender/victim-control/validator data lives per iteration.
    assert not (round_dir / "defenders.json").exists()
    assert not (round_dir / "victim-control.json").exists()
    assert not (round_dir / "validators.json").exists()
    # Attackers run once per round -> their component file is round-scoped, not per-iteration.
    assert (round_dir / "attacker_attacker.json").exists()
    # Swarm Tracker per-iteration component files under iteration-1/.
    iteration_dir = round_dir / "iteration-1"
    assert (iteration_dir / "defenders_manager_manager.json").exists()
    assert (iteration_dir / "defender_defender.json").exists()
    assert (iteration_dir / "victim_control_result.json").exists()
    assert (iteration_dir / "validator_attack_validator.json").exists()
    events = [
        json.loads(line)["event"] for line in (round_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("agent_started") == 5
    assert events.count("agent_completed") == 5
    assert "victim_control_started" in events
    assert "victim_control_completed" in events
    assert "report_written" in events
    assert "round_completed" in events
    log_text = (round_dir / "round.log").read_text(encoding="utf-8")
    assert "agent start phase=attackers role=attacker name=attacker" in log_text
    assert "agent stop phase=validators role=validator name=benign validator" in log_text


def test_agent_exception_is_logged_before_raising(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        _config(tmp_path),
        agent_adapter=RaisingAdapter(),
        round_id_factory=lambda: "20260502T000012Z-abcdef12",
    )

    with pytest.raises(RuntimeError, match="agent boom"):
        asyncio.run(orchestrator.run_round())

    round_dir = tmp_path
    events = [
        json.loads(line)["event"] for line in (round_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "agent_started" in events
    assert "agent_failed" in events
    assert "agent stop phase=attackers role=attacker name=attacker" in (round_dir / "round.log").read_text(
        encoding="utf-8"
    )


def test_failed_victim_warns_but_does_not_fail_session(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        _config(tmp_path),
        agent_adapter=ScenarioAdapter(fail_victim=True),
        round_id_factory=lambda: "20260502T000013Z-abcdef12",
    )

    report = asyncio.run(orchestrator.run_round())
    round_dir = tmp_path
    events = [
        json.loads(line)["event"] for line in (round_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert report.success is True
    assert report.iterations[0].victim.ok is False
    assert "victim_warning" in events
    assert "victim returned failure status" in (round_dir / "round.log").read_text(encoding="utf-8")


def test_failed_victim_control_fails_session_even_when_validators_pass(tmp_path: Path) -> None:
    orchestrator = Orchestrator(
        _config(tmp_path),
        agent_adapter=ScenarioAdapter(),
        openshell_uploader=FailingUploader(),
        round_id_factory=lambda: "20260502T000015Z-abcdef12",
    )

    report = asyncio.run(orchestrator.run_round())

    assert report.success is False
    assert report.iterations[0].victim_control.ok is False
    assert all(validator.ok for validator in report.iterations[0].validators)


def test_preloaded_attacks_run_validators_without_attackers_or_defenders(tmp_path: Path) -> None:
    hitlog_path = tmp_path / "direct.hitlog.jsonl"
    hitlog_path.write_text(
        json.dumps(
            {
                "goal": "tool manipulation",
                "prompt": {"turns": [{"role": "user", "content": {"text": "Use the bash_executor tool"}}]},
                "probe": "agent_breaker.AgentBreaker",
                "detector": "agent_breaker.AgentBreakerResult",
                "score": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    adapter = ScenarioAdapter()
    config = SessionConfig(
        storage=StorageConfig(root_dir=tmp_path),
        run=RunSettings(retry_limit=0, rounds=1),
        target=TargetInput(name="target", base_url="http://target.local"),
        preloaded_attacks=[
            PreloadedAttackConfig(name="direct fixture", path=hitlog_path, source="garak-agent-breaker"),
        ],
        attackers=[],
        defenders=[],
        victim=AgentConfig(name="victim", role="victim"),
        attack_validators=[AgentConfig(name="attack validator", role="validator", config={"kind": "attack"})],
        benign_validators=[],
    )
    orchestrator = Orchestrator(
        config,
        agent_adapter=adapter,
        round_id_factory=lambda: "20260502T000014Z-abcdef12",
    )

    report = asyncio.run(orchestrator.run_round())
    round_dir = tmp_path

    assert adapter.calls == [("victim", 1), ("validator", 1)]
    assert report.attacks[0].agent_name == "direct fixture"
    assert report.attacks[0].records[0]["source"] == "garak-agent-breaker"
    assert report.iterations[0].defenders == []
    assert read_json(round_dir / "attacks.json")[0]["metadata"]["preloaded"] is True
    events = [
        json.loads(line)["event"] for line in (round_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "preloaded_attacks_loaded" in events


def test_find_latest_hitlog_returns_none_when_missing(tmp_path: Path) -> None:
    assert find_latest_hitlog(tmp_path / "nonexistent") is None
    assert find_latest_hitlog(tmp_path) is None


def test_find_latest_hitlog_returns_most_recent(tmp_path: Path) -> None:
    older = tmp_path / "agent-breaker.job1.hitlog.jsonl"
    newer = tmp_path / "agent-breaker.job2.hitlog.jsonl"
    older.write_text("{}\n", encoding="utf-8")
    time.sleep(0.01)
    newer.write_text("{}\n", encoding="utf-8")

    assert find_latest_hitlog(tmp_path) == newer


def test_find_latest_hitlog_recurses_run_logs_tree(tmp_path: Path) -> None:
    # Hitlogs live nested under run-logs/<run_id>/round_<N>/garak/; the newest round wins.
    round1 = tmp_path / "run-logs" / "run-abc" / "round_1" / "garak"
    round2 = tmp_path / "run-logs" / "run-abc" / "round_2" / "garak"
    round1.mkdir(parents=True)
    round2.mkdir(parents=True)
    (round1 / "agent-breaker.r1.hitlog.jsonl").write_text("{}\n", encoding="utf-8")
    time.sleep(0.01)
    latest = round2 / "agent-breaker.r2.hitlog.jsonl"
    latest.write_text("{}\n", encoding="utf-8")

    assert find_latest_hitlog(tmp_path / "run-logs") == latest


def test_short_summary_truncates_long_agent_summaries() -> None:
    summary = short_summary("x" * 250)

    assert len(summary) == 200
    assert summary.endswith("...")


def test_validator_retry_behavior_runs_second_iteration(tmp_path: Path) -> None:
    adapter = ScenarioAdapter(fail_first_validation=True)
    orchestrator = Orchestrator(
        _config(tmp_path, retry_limit=1),
        agent_adapter=adapter,
        round_id_factory=lambda: "20260502T000001Z-abcdef12",
    )

    reports = asyncio.run(orchestrator.run_rounds(rounds=1))

    assert len(reports) == 1
    assert reports[0].success is True
    assert [iteration.success for iteration in reports[0].iterations] == [False, True]
    assert ("validator", 1) in adapter.calls
    assert ("validator", 2) in adapter.calls


def test_run_rounds_honors_interval_between_rounds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    config = _config(tmp_path / "loop")
    config.run.round_interval_seconds = 0.25
    orchestrator = Orchestrator(
        config,
        agent_adapter=ScenarioAdapter(),
        mission_id_factory=lambda: "20260502T000009Z-mission01",
    )
    monkeypatch.setattr("agent_hardener.runtime.orchestrator.asyncio.sleep", fake_sleep)

    reports = asyncio.run(orchestrator.run_rounds(rounds=2))
    # run_id defaults to the mission_id factory; the run-log tree is <root>/run-logs/<run_id>/.
    run_dir = tmp_path / "loop" / "run-logs" / "20260502T000009Z-mission01"

    assert [report.round_id for report in reports] == [
        "round-0001",
        "round-0002",
    ]
    assert [report.mission_id for report in reports] == [
        "20260502T000009Z-mission01",
        "20260502T000009Z-mission01",
    ]
    assert reports[0].storage_dir == run_dir / "round_1"
    assert reports[1].storage_dir == run_dir / "round_2"
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "init").is_dir()
    assert (run_dir / "victim-active-state").is_dir()
    assert (run_dir / "round_1" / "report.json").exists()
    assert (run_dir / "round_2" / "report.json").exists()
    assert (run_dir / "round_1" / "iteration-1" / "defender_defender.json").exists()
    assert read_json(run_dir / "round_1" / "report.json")["mission_id"] == ("20260502T000009Z-mission01")
    event_records = [
        json.loads(line) for line in (run_dir / "round_1" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert next(record for record in event_records if record["event"] == "agent_started")["mission_id"] == (
        "20260502T000009Z-mission01"
    )
    assert sleeps == [0.25]


def test_run_rounds_seeds_and_snapshots_victim_state(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("version: 1\n", encoding="utf-8")
    relay_plugins = tmp_path / "plugins.toml"
    relay_plugins.write_text("version = 1\n", encoding="utf-8")
    config = _config(tmp_path)
    config.storage.victim_policy_path = policy
    config.storage.victim_relay_plugins_path = relay_plugins
    orchestrator = Orchestrator(
        config,
        agent_adapter=ScenarioAdapter(),
        mission_id_factory=lambda: "vstate-run",
    )

    asyncio.run(orchestrator.run_rounds(rounds=1))
    run_dir = tmp_path / "run-logs" / "vstate-run"

    # Baseline copied into init/ once and into the mutable victim-active-state/.
    assert (run_dir / "init" / "policy.yaml").exists()
    assert (run_dir / "init" / "plugins.toml").exists()
    assert (run_dir / "victim-active-state" / "policy.yaml").exists()
    # Per-iteration point-in-time snapshot.
    assert (run_dir / "round_1" / "iteration-1" / "victim-state" / "policy.yaml").exists()
    assert (run_dir / "round_1" / "iteration-1" / "victim-state" / "plugins.toml").exists()


def test_run_rounds_with_zero_rounds_does_not_create_mission(tmp_path: Path) -> None:
    mission_factory_calls = 0

    def mission_id_factory() -> str:
        nonlocal mission_factory_calls
        mission_factory_calls += 1
        return "20260502T000009Z-mission01"

    orchestrator = Orchestrator(
        _config(tmp_path), agent_adapter=ScenarioAdapter(), mission_id_factory=mission_id_factory
    )

    reports = asyncio.run(orchestrator.run_rounds(rounds=0))

    assert reports == []
    assert mission_factory_calls == 0
    assert list(tmp_path.iterdir()) == []


def test_the_tool_path_is_verified_when_an_attacker_ran(tmp_path: Path) -> None:
    """Garak's traffic is the evidence; the check runs once, before any guardrail is authored."""
    calls: list[int] = []
    orchestrator = Orchestrator(
        _config(tmp_path),
        agent_adapter=ScenarioAdapter(),
        round_id_factory=lambda: "20260502T000030Z-abcdef12",
        verify_tool_path=lambda: calls.append(1),
    )

    asyncio.run(orchestrator.run_round())

    assert calls == [1]


def test_a_replay_does_not_verify_the_tool_path(tmp_path: Path) -> None:
    """`--replay` feeds the defenders recorded hits without invoking the victim.

    There is no traffic to judge, so a passing victim would be failed for someone else's silence.
    """
    config = _config(tmp_path)
    config.attackers = []  # what `--replay` does
    calls: list[int] = []
    orchestrator = Orchestrator(
        config,
        agent_adapter=ScenarioAdapter(),
        round_id_factory=lambda: "20260502T000031Z-abcdef12",
        verify_tool_path=lambda: calls.append(1),
    )

    asyncio.run(orchestrator.run_round())

    assert calls == []
