# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the attack stage's hard-fail on a non-completing attacker."""

from __future__ import annotations

import pytest

from agent_hardener.errors import AttackerError
from agent_hardener.models.contracts import AttackRecord
from agent_hardener.runtime.stages.attack import raise_on_attacker_failure

pytestmark = pytest.mark.unit


class _Agent:
    def __init__(self, name: str) -> None:
        self.name = name


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
