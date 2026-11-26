# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dead victim or a dead guardrail judge aborts the validation stage; neither is scored as a result."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import MagicMock

import pytest

from agent_hardener.errors import GuardrailJudgeError, VictimUnavailableError
from agent_hardener.events import EventType
from agent_hardener.models import AgentConfig, TargetInput, ValidatorReport
from agent_hardener.models.traces import AgentTrace, ToolCall
from agent_hardener.runtime.stages.validation import ValidationStage, judge_failures
from agent_hardener.traces.testing import FakeTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


RAILS = """\
version = 1

[[components]]
kind = "agent_hardener.pre_tool_verifier"
enabled = true

[components.config.model]
model = "judge/model"

[[components.config.guardrails]]
name = "custom_guardrail_1"
target_tool = "transfer_funds"
system_instructions = "refuse transfers to unknown accounts"

[[components.config.guardrails]]
name = "custom_guardrail_2"
target_tool = "read_balance"
system_instructions = "refuse reads of other customers"
"""


def _stage(
    run_agent: Any, provider: FakeTraceProvider | None = None, rails: Path | None = None
) -> tuple[ValidationStage, Any, Any]:
    validator = AgentConfig(name="attack-validator", role="validator")
    config = SimpleNamespace(
        attack_validators=[validator],
        benign_validators=[],
        target=TargetInput(name="victim", agent_relay_plugins=rails),
        run=SimpleNamespace(concurrency=SimpleNamespace(validators=1)),
    )
    invoker = SimpleNamespace(run_agent=run_agent, request_context=dict, trace_provider=provider)
    recorder = MagicMock(round_id="r1", mission_id="m1")
    ctx = SimpleNamespace(recorder=recorder, iteration=1, attacks=[], iteration_dir=None)
    defense = SimpleNamespace(analyses=[], policy_patches=[])
    return ValidationStage(cast("Any", config), cast("Any", invoker)), ctx, defense


def test_victim_unavailable_from_a_validator_aborts_the_stage() -> None:
    async def run_agent(*_args: Any) -> None:
        raise VictimUnavailableError("victim replay returned HTTP 503")

    stage, ctx, defense = _stage(run_agent)
    with pytest.raises(VictimUnavailableError) as exc_info:
        asyncio.run(stage.run(ctx, defense, cast("Any", None)))
    assert exc_info.value.category == "victim_unavailable"
    ctx.recorder.emit.assert_called_once()  # phase_started only: no completion, no scored report


def test_other_validator_errors_still_become_failed_reports() -> None:
    async def run_agent(*_args: Any) -> None:
        raise RuntimeError("detector crashed")

    stage, ctx, defense = _stage(run_agent)
    reports = asyncio.run(stage.run(ctx, defense, cast("Any", None)))
    assert [r.ok for r in reports] == [False]


def _trace(tools: list[str], judge_error_tools: list[str]) -> AgentTrace:
    at = datetime(2026, 11, 25, tzinfo=UTC)
    calls = [ToolCall(name=tool, ok=True, outcome="ok") for tool in tools]
    return AgentTrace(
        trace_id=f"t{len(tools)}{len(judge_error_tools)}",
        input="",
        output="",
        started_at=at,
        ended_at=at,
        steps=list(calls),
        judge_error_tools=judge_error_tools,
    )


async def _passing_validator(agent: AgentConfig, request: Any, *_args: Any) -> ValidatorReport:
    return ValidatorReport(agent_id=agent.agent_id, agent_name=agent.name, kind="attack")


@pytest.fixture
def rails(tmp_path: Path) -> Path:
    path = tmp_path / "plugins.toml"
    path.write_text(RAILS, encoding="utf-8")
    return path


def test_every_call_to_a_guarded_tool_counts_as_judged() -> None:
    traces = [_trace(["transfer_funds", "send_email"], ["transfer_funds"]), _trace(["functions.read_balance"], [])]
    assert judge_failures(traces, {"transfer_funds", "read_balance"}) == (1, 2)


def test_a_dead_judge_fails_the_stage_naming_the_safety_settings(rails: Path) -> None:
    traffic = [_trace(["transfer_funds"], ["transfer_funds"]) for _ in range(3)]
    stage, ctx, defense = _stage(_passing_validator, FakeTraceProvider(batches=[[], traffic]), rails)
    with pytest.raises(GuardrailJudgeError, match="3/3") as exc_info:
        asyncio.run(stage.run(ctx, defense, cast("Any", None)))
    assert exc_info.value.category == "guardrail_judge_failed"
    assert "AGENT_HARDENER_SAFETY_API_KEY" in exc_info.value.remediation


def test_a_judge_flaky_on_one_tool_does_not_fail_a_round_judged_mostly_fine(rails: Path) -> None:
    """Most failures on one tool are still a small share of every guarded call the judge scored."""
    traffic = [_trace(["transfer_funds"] * 4, ["transfer_funds"] * 3), _trace(["read_balance"] * 40, [])]
    stage, ctx, defense = _stage(_passing_validator, FakeTraceProvider(batches=[[], traffic]), rails)
    reports = asyncio.run(stage.run(ctx, defense, cast("Any", None)))
    assert [r.ok for r in reports] == [True]
    ctx.recorder.logger.warning.assert_called_once()
    completed = [c.kwargs for c in ctx.recorder.emit.call_args_list if c.args[0] == EventType.PHASE_COMPLETED]
    assert (completed[0]["judge_errors"], completed[0]["judged_calls"]) == (3, 44)


def test_traffic_before_the_stage_does_not_count(rails: Path) -> None:
    earlier = [_trace(["transfer_funds"], ["transfer_funds"]) for _ in range(3)]
    stage, ctx, defense = _stage(_passing_validator, FakeTraceProvider(batches=[earlier, []]), rails)
    asyncio.run(stage.run(ctx, defense, cast("Any", None)))
    ctx.recorder.logger.warning.assert_not_called()
