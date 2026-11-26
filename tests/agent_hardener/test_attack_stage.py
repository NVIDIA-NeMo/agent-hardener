# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the attack stage: its hard-fail on a non-completing attacker, optional attackers, and its cap."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.errors import AttackerError
from agent_hardener.events import EventType
from agent_hardener.models import AgentConfig, RunConcurrencySettings, TargetInput
from agent_hardener.models.contracts import AttackRecord
from agent_hardener.runtime.stages.attack import AttackStage, raise_on_attacker_failure

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class _Agent:
    def __init__(self, name: str, *, required: bool = True) -> None:
        self.name = name
        self.required = required


def _record(ok: bool, error: str | None = None) -> AttackRecord:
    return AttackRecord(agent_id="a1", agent_name="garak-agent-breaker", ok=ok, error=error)


def test_raises_when_a_live_attacker_times_out() -> None:
    with pytest.raises(AttackerError) as exc_info:
        raise_on_attacker_failure([_Agent("garak-agent-breaker")], [_record(ok=False, error="TimeoutError")])
    assert exc_info.value.category == "attacker_failed"
    assert "garak-agent-breaker" in str(exc_info.value)
    assert "TimeoutError" in str(exc_info.value)


def test_no_raise_when_attacker_completes_even_with_zero_hits() -> None:
    # ok=True with no records is a legitimate "found nothing" result — must NOT be treated as a failure.
    raise_on_attacker_failure([_Agent("garak-agent-breaker")], [_record(ok=True)])


def test_no_raise_for_replay_run_with_no_live_attackers() -> None:
    raise_on_attacker_failure([], [])


def test_no_raise_when_only_an_optional_attacker_fails() -> None:
    raise_on_attacker_failure(
        [_Agent("garak-agent-breaker"), _Agent("garak-tool-chain", required=False)],
        [_record(ok=True), _record(ok=False, error="boom")],
    )


def test_a_required_failure_still_raises_beside_an_optional_one() -> None:
    with pytest.raises(AttackerError) as exc_info:
        raise_on_attacker_failure(
            [_Agent("garak-agent-breaker"), _Agent("garak-tool-chain", required=False)],
            [_record(ok=False, error="TimeoutError"), _record(ok=False, error="boom")],
        )
    assert "garak-agent-breaker" in str(exc_info.value)
    assert "garak-tool-chain" not in str(exc_info.value)


class _Recorder:
    round_id = "round_001"
    mission_id = "mission"
    events = None

    def __init__(self) -> None:
        self.logger = logging.getLogger("test.attack_stage")
        self.emitted: list[tuple[Any, dict[str, Any]]] = []

    def emit(self, event: Any, **payload: Any) -> None:
        self.emitted.append((event, payload))


class _Store:
    def __init__(self, root: Path) -> None:
        self.round_dir = root

    def round_artifact_dir(self, _kind: Any) -> Path:
        return self.round_dir / "garak"

    def write_round_json(self, name: str, _outputs: Any) -> Path:
        return self.round_dir / name

    def write_component(self, *_args: Any) -> None:
        pass


class _Invoker:
    """Runs each attacker after a short wait, recording how many ran at once."""

    def __init__(self, failing: set[str]) -> None:
        self.failing = failing
        self.running = 0
        self.peak = 0

    def request_context(self) -> dict[str, Any]:
        return {}

    async def run_agent(self, agent: AgentConfig, *_args: Any) -> AttackRecord:
        self.running += 1
        self.peak = max(self.peak, self.running)
        await asyncio.sleep(0.01)
        self.running -= 1
        if agent.name in self.failing:
            raise RuntimeError(f"{agent.name} crashed")
        return AttackRecord(agent_id=agent.agent_id, agent_name=agent.name, ok=True)


def _attacker(name: str, *, required: bool = True) -> AgentConfig:
    return AgentConfig(name=name, role="attacker", implementation="tests:run", required=required)


def _stage(attackers: list[AgentConfig], invoker: _Invoker, cap: int | None) -> AttackStage:
    config = SimpleNamespace(
        preloaded_attacks=[],
        target=TargetInput(name="victim", base_url="http://127.0.0.1:8000/v1/chat/completions"),
        attackers=attackers,
        run=SimpleNamespace(concurrency=RunConcurrencySettings(attackers=cap)),
    )
    return AttackStage(config, invoker)  # type: ignore[arg-type]


@pytest.mark.parametrize(("cap", "peak"), [(None, 3), (1, 1), (2, 2)])
def test_attackers_run_at_most_the_configured_number_at_once(tmp_path: Path, cap: int | None, peak: int) -> None:
    invoker = _Invoker(failing=set())
    stage = _stage([_attacker(f"a{i}") for i in range(3)], invoker, cap)
    asyncio.run(stage.run(_Recorder(), _Store(tmp_path)))  # type: ignore[arg-type]
    assert invoker.peak == peak


def test_an_optional_attacker_failure_is_reported_and_the_round_goes_on(tmp_path: Path) -> None:
    recorder = _Recorder()
    stage = _stage([_attacker("main"), _attacker("extra", required=False)], _Invoker(failing={"extra"}), None)
    outputs = asyncio.run(stage.run(recorder, _Store(tmp_path)))  # type: ignore[arg-type]
    assert [(out.agent_name, out.ok) for out in outputs] == [("main", True), ("extra", False)]
    tolerated = [payload for event, payload in recorder.emitted if event == EventType.ATTACKER_FAILURE_TOLERATED]
    assert len(tolerated) == 1
    assert tolerated[0]["agent_name"] == "extra"
    assert "extra crashed" in tolerated[0]["error"]


def test_a_required_attacker_failure_still_fails_the_round(tmp_path: Path) -> None:
    stage = _stage([_attacker("main"), _attacker("extra", required=False)], _Invoker(failing={"main"}), None)
    with pytest.raises(AttackerError, match="main"):
        asyncio.run(stage.run(_Recorder(), _Store(tmp_path)))  # type: ignore[arg-type]


def test_required_and_the_concurrency_cap_do_not_change_agent_ids() -> None:
    assert _attacker("a").agent_id == _attacker("a", required=False).agent_id
