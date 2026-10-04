# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Round and iteration report models written alongside the run logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from agent_hardener.models.base import AgentHardenerModel
from agent_hardener.models.contracts import (
    AttackRecord,
    DefenderAnalysis,
    ValidatorReport,
    VictimControlResult,
    VictimResult,
)


class RoundIterationReport(AgentHardenerModel):
    """Report for one defender/policy/victim/validator iteration."""

    iteration: int = Field(ge=1)
    defenders: list[DefenderAnalysis] = Field(default_factory=list)
    policy_patches: list[dict[str, Any]] = Field(default_factory=list)
    victim_control: VictimControlResult
    victim: VictimResult
    validators: list[ValidatorReport] = Field(default_factory=list)
    success: bool


class RoundReport(AgentHardenerModel):
    """Final round report written separately from logs."""

    round_id: str
    mission_id: str | None = None
    started_at: datetime
    ended_at: datetime
    success: bool
    attacks: list[AttackRecord] = Field(default_factory=list)
    iterations: list[RoundIterationReport] = Field(default_factory=list)
    storage_dir: Path
