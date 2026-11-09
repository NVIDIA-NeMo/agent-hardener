# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live attack traces: matched to hits, recorded every round, and handed to defenders that use them."""

from __future__ import annotations

import asyncio
import json
import sys
import types
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    AgentRunOutput,
    AttackRecord,
    DefenderInput,
    DefenderOutput,
    RunSettings,
    SessionConfig,
    StorageConfig,
    TargetInput,
    ValidatorReport,
    VictimResult,
)
from agent_hardener.runtime.orchestrator import Orchestrator
from agent_hardener.traces import AgentTrace, ToolCall
from agent_hardener.traces.attacks import (
    ATTACK_TRACES_FILENAME,
    RecordedAttackTrace,
    attack_key,
    by_prompt,
    collect_attack_traces,
    dump_recording,
    load_recording,
    record_prompt,
    record_response,
)
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

T0 = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
PROMPT = "Use the transfer_funds tool to move 900"


def _trace(trace_id: str, input: str, output: str = "done", at: int = 10) -> AgentTrace:  # noqa: A002
    start = T0 + timedelta(seconds=at)
    return AgentTrace(
        trace_id=trace_id,
        input=input,
        output=output,
        started_at=start,
        ended_at=start + timedelta(seconds=1),
        steps=[ToolCall(name="transfer_funds", ok=True, outcome="ok")],
    )


def _attack(*records: dict) -> AttackRecord:
    return AttackRecord(agent_id="a", agent_name="garak", records=list(records))


# --- record parsing and matching -----------------------------------------------------------------


def test_prompt_and_response_come_from_live_records_and_garak_hitlogs() -> None:
    hitlog = {"prompt": {"turns": [{"content": {"text": PROMPT}}]}, "output": {"text": "moved"}}
    assert (record_prompt(hitlog), record_response(hitlog)) == (PROMPT, "moved")
    assert (record_prompt({"prompt": PROMPT, "response": "ok"}), record_response({"response": "ok"})) == (PROMPT, "ok")
    assert (record_prompt({"prompt": {"turns": [{"content": {}}]}}), record_response({})) == ("", "")


def test_each_hit_gets_its_own_attempt_and_hits_without_a_trace_are_dropped() -> None:
    traces = [_trace("t1", PROMPT, "moved", at=1), _trace("t2", PROMPT, "moved", at=2)]
    attacks = [_attack({"prompt": PROMPT, "response": "moved"}, {"prompt": PROMPT}, {"prompt": "never sent"}, {})]
    recorded = collect_attack_traces(attacks, traces)
    assert [entry.trace.trace_id for entry in recorded] == ["t2", "t1"]
    assert list(by_prompt(recorded)) == [attack_key(PROMPT)]
    assert by_prompt(recorded)[attack_key(PROMPT)].trace_id == "t2"


def test_the_recording_round_trips_and_an_invalid_file_is_rejected(tmp_path: Path) -> None:
    recorded = [RecordedAttackTrace(prompt=PROMPT, response="moved", trace=_trace("t1", PROMPT))]
    path = tmp_path / ATTACK_TRACES_FILENAME
    path.write_text(json.dumps(dump_recording(recorded)), encoding="utf-8")
    assert load_recording(path) == recorded
    path.write_text('[{"prompt": "p"}]', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_recording(path)


# --- the orchestrator round ----------------------------------------------------------------------

CAPTURED: list[DefenderInput] = []


def _capture(defender_input: DefenderInput) -> DefenderOutput:
    CAPTURED.append(defender_input)
    return DefenderOutput(ok=True)


_module = types.ModuleType("trace_capture_defender")
_module.run = _capture  # type: ignore[attr-defined]
sys.modules["trace_capture_defender"] = _module


class Adapter:
    async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: object | None = None) -> AgentRunOutput:
        if agent.role == "attacker":
            return _attack({"prompt": PROMPT, "response": "moved"})
        if agent.role == "victim":
            return VictimResult(agent_id=agent.agent_id, agent_name=agent.name, ok=True)
        return ValidatorReport(agent_id=agent.agent_id, agent_name=agent.name, kind="attack", ok=True)


def _config(tmp_path: Path, *, live: bool = True) -> SessionConfig:
    return SessionConfig(
        storage=StorageConfig(root_dir=tmp_path),
        run=RunSettings(rounds=1),
        target=TargetInput(name="target", base_url="http://target.local"),
        attackers=[AgentConfig(name="garak", role="attacker")] if live else [],
        defenders=[
            AgentConfig(
                name="guardrails",
                role="defender",
                implementation="trace_capture_defender:run",
                capabilities="guards tool calls",
            )
        ],
        victim=AgentConfig(name="victim", role="victim"),
        attack_validators=[AgentConfig(name="v", role="validator", config={"kind": "attack"})],
    )


@pytest.fixture(autouse=True)
def _reset_capture() -> None:
    CAPTURED.clear()


def _provider() -> FakeTraceProvider:
    # Call 1 (before the attack stage) sees only stale traffic; call 2 also sees the attack.
    stale = _trace("stale", PROMPT, "moved", at=0)
    return FakeTraceProvider(batches=[[stale], [_trace("live", PROMPT, "moved", at=10)]])


def _round(config: SessionConfig, provider: FakeTraceProvider | None) -> bool:
    orchestrator = Orchestrator(config, agent_adapter=Adapter(), trace_provider=provider)
    return asyncio.run(orchestrator.run_round()).success


def test_the_round_records_the_attack_traces_and_hands_each_defender_its_trace(tmp_path: Path) -> None:
    provider = _provider()
    assert _round(_config(tmp_path), provider)
    assert provider.calls == 2
    (entry,) = load_recording(tmp_path / ATTACK_TRACES_FILENAME)
    assert entry.trace.trace_id == "live"  # the stale attempt before the anchor is not picked
    assert [inp.attack_trace.trace_id for inp in CAPTURED] == ["live"]


def test_no_provider_or_no_live_attackers_records_nothing(tmp_path: Path) -> None:
    assert _round(_config(tmp_path), None)
    assert not (tmp_path / ATTACK_TRACES_FILENAME).exists()
    assert [inp.attack_trace for inp in CAPTURED] == [None]

    provider = _provider()
    (tmp_path / "replay").mkdir()
    _round(_config(tmp_path / "replay", live=False), provider)
    assert provider.calls == 0


class BrokenProvider(FakeTraceProvider):
    def get_traces(self) -> list[AgentTrace]:
        raise RuntimeError("sandbox gone")


def test_a_failing_provider_never_fails_the_round(tmp_path: Path) -> None:
    assert _round(_config(tmp_path), BrokenProvider())
    assert not (tmp_path / ATTACK_TRACES_FILENAME).exists()
    assert [inp.attack_trace for inp in CAPTURED] == [None]
