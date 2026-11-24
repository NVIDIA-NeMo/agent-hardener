# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A dead victim aborts the validation stage; it is never scored as a failed validator."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from agent_hardener.errors import VictimUnavailableError
from agent_hardener.models import AgentConfig, TargetInput
from agent_hardener.runtime.stages.validation import ValidationStage

pytestmark = pytest.mark.unit


def _stage(run_agent: Any) -> tuple[ValidationStage, Any, Any]:
    validator = AgentConfig(name="attack-validator", role="validator")
    config = SimpleNamespace(
        attack_validators=[validator],
        benign_validators=[],
        target=TargetInput(name="victim"),
        run=SimpleNamespace(concurrency=SimpleNamespace(validators=1)),
    )
    invoker = SimpleNamespace(run_agent=run_agent, request_context=dict)
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
