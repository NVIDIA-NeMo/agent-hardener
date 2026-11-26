# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A round whose defenders all failed stops before deploy and validation, saying why."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from agent_hardener.errors import DefendersFailedError, DefendersNotApplicableError
from agent_hardener.models import AttackRecord, DefendersManagerOutput
from agent_hardener.runtime.stages.defense import _require_mitigation

_ATTACK = AttackRecord(agent_id="a", agent_name="garak")


def _ctx(*, attacks: list[AttackRecord], feedback: dict[str, Any] | None = None) -> Any:
    return cast("Any", SimpleNamespace(attacks=attacks, validation_feedback=feedback))


def test_no_successful_defender_fails_with_each_distinct_error_once() -> None:
    failed = DefendersManagerOutput(
        ok=False, error="Agent guard failed: 401 | Agent guard failed: 401 | Agent policy failed: timeout"
    )
    with pytest.raises(DefendersFailedError) as exc_info:
        _require_mitigation(failed, _ctx(attacks=[_ATTACK]))
    assert str(exc_info.value) == (
        "no defender produced a mitigation: Agent guard failed: 401 | Agent policy failed: timeout"
    )
    assert exc_info.value.category == "defenders_failed"


def test_a_round_where_every_defender_abstained_is_out_of_scope() -> None:
    abstained = DefendersManagerOutput(ok=False, not_applicable=["Agent policy did not apply: out_of_scope: no host"])
    with pytest.raises(DefendersNotApplicableError, match=r"no defender applies.*out_of_scope: no host"):
        _require_mitigation(abstained, _ctx(attacks=[_ATTACK]))


def test_one_failed_defender_fails_the_round_even_if_the_rest_abstained() -> None:
    mixed = DefendersManagerOutput(
        ok=False, error="Agent guard failed: 401", not_applicable=["Agent policy did not apply: out_of_scope"]
    )
    with pytest.raises(DefendersFailedError, match="Agent guard failed: 401"):
        _require_mitigation(mixed, _ctx(attacks=[_ATTACK]))


@pytest.mark.parametrize(
    ("output", "ctx"),
    [
        (DefendersManagerOutput(ok=True), _ctx(attacks=[_ATTACK])),
        (DefendersManagerOutput(ok=False, error="nothing routed"), _ctx(attacks=[])),
        (DefendersManagerOutput(ok=False, error="nothing to retry"), _ctx(attacks=[_ATTACK], feedback={"g": {}})),
    ],
    ids=["a-defender-succeeded", "no-attacks", "retry"],
)
def test_the_round_continues(output: DefendersManagerOutput, ctx: Any) -> None:
    _require_mitigation(output, ctx)
