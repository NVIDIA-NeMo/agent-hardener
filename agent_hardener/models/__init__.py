# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic contracts for Agent Hardener platform components.

The models are grouped by concern across submodules (``base``, ``agent``, ``contracts``,
``infra``, ``session``, ``reports``) and re-exported here, so ``from agent_hardener.models import X``
keeps working for every public model, literal, and constant.
"""

from __future__ import annotations

from agent_hardener.models.agent import AgentConfig, TargetInput
from agent_hardener.models.base import AgentHardenerModel, AgentRole, ValidatorKind, VictimControlType
from agent_hardener.models.contracts import (
    ARTIFACT_DIR_KEY,
    CURRENT_POLICY_KEY,
    AgentRunInput,
    AgentRunOutput,
    Artifact,
    AttackRecord,
    BenignRequest,
    DefenderAnalysis,
    DefenderInput,
    DefenderOutput,
    DefendersManagerInput,
    DefendersManagerOutput,
    ValidationFeedback,
    ValidatorReport,
    VictimControlResult,
    VictimResult,
)
from agent_hardener.models.infra import (
    BackendEndpoint,
    BackendServiceSpec,
    DefenderSettings,
    GarakSettings,
    PreloadedAttackConfig,
    RelayVictimSpec,
    VictimControlConfig,
)
from agent_hardener.models.reports import RoundIterationReport, RoundReport
from agent_hardener.models.session import RunConcurrencySettings, RunSettings, SessionConfig, StorageConfig

__all__ = [
    "ARTIFACT_DIR_KEY",
    "CURRENT_POLICY_KEY",
    "AgentConfig",
    "AgentHardenerModel",
    "AgentRole",
    "AgentRunInput",
    "AgentRunOutput",
    "Artifact",
    "AttackRecord",
    "BackendEndpoint",
    "BackendServiceSpec",
    "BenignRequest",
    "DefenderAnalysis",
    "DefenderInput",
    "DefenderOutput",
    "DefenderSettings",
    "DefendersManagerInput",
    "DefendersManagerOutput",
    "GarakSettings",
    "PreloadedAttackConfig",
    "RelayVictimSpec",
    "RoundIterationReport",
    "RoundReport",
    "RunConcurrencySettings",
    "RunSettings",
    "SessionConfig",
    "StorageConfig",
    "TargetInput",
    "ValidationFeedback",
    "ValidatorKind",
    "ValidatorReport",
    "VictimControlConfig",
    "VictimControlResult",
    "VictimResult",
]
