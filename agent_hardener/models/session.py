# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Top-level orchestrator session configuration and run/retry settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, ValidationInfo, field_validator, model_validator

from agent_hardener.models.agent import AgentConfig, TargetInput
from agent_hardener.models.base import AgentHardenerModel, AgentRole
from agent_hardener.models.infra import GarakSettings, PreloadedAttackConfig, VictimControlConfig


class StorageConfig(AgentHardenerModel):
    """Storage settings for orchestration output."""

    root_dir: Path
    run_id: str | None = None
    round_number: int | None = None
    victim_policy_path: Path | None = None
    victim_relay_plugins_path: Path | None = None


class RunConcurrencySettings(AgentHardenerModel):
    """Concurrency caps for orchestration phases and per-agent item work."""

    defenders: int = Field(default=2, ge=1)
    validators: int = Field(default=2, ge=1)
    guardrails_findings: int = Field(default=2, ge=1)
    openshell_policy_findings: int = Field(default=2, ge=1)
    benign_validator_rows: int = Field(default=8, ge=1)
    attack_validator_hits: int = Field(default=6, ge=1)


class RunSettings(AgentHardenerModel):
    """Run-loop and retry settings."""

    retry_limit: int = Field(default=0, ge=0)
    rounds: int = Field(default=500, ge=1)
    round_interval_seconds: float = Field(default=0.0, ge=0)
    concurrency: RunConcurrencySettings = Field(default_factory=RunConcurrencySettings)


class SessionConfig(AgentHardenerModel):
    """Full orchestrator configuration loaded from JSON or YAML."""

    storage: StorageConfig
    run: RunSettings = Field(default_factory=RunSettings)
    victim_control: VictimControlConfig = Field(default_factory=VictimControlConfig)
    garak: GarakSettings | None = None
    target: TargetInput
    context: dict[str, Any] = Field(default_factory=dict)
    # Explicit benign-suite CSV to use as-is (skips synthesis). Set by `run --benign-suite <path>`.
    benign_suite_path: Path | None = None
    preloaded_attacks: list[PreloadedAttackConfig] = Field(default_factory=list)
    attackers: list[AgentConfig] = Field(default_factory=list)
    defenders: list[AgentConfig] = Field(default_factory=list)
    victim: AgentConfig
    attack_validators: list[AgentConfig] = Field(default_factory=list)
    benign_validators: list[AgentConfig] = Field(default_factory=list)

    @field_validator("attackers")
    @classmethod
    def validate_attackers(cls, value: list[AgentConfig]) -> list[AgentConfig]:
        return _validate_agent_roles(value, "attacker", "attackers")

    @field_validator("defenders")
    @classmethod
    def validate_defenders(cls, value: list[AgentConfig]) -> list[AgentConfig]:
        return _validate_agent_roles(value, "defender", "defenders")

    @field_validator("victim")
    @classmethod
    def validate_victim(cls, value: AgentConfig) -> AgentConfig:
        if value.role != "victim":
            msg = "victim must have role 'victim'"
            raise ValueError(msg)
        return value

    @field_validator("attack_validators")
    @classmethod
    def validate_attack_validators(cls, value: list[AgentConfig]) -> list[AgentConfig]:
        return _validate_agent_roles(value, "validator", "attack_validators")

    @field_validator("benign_validators")
    @classmethod
    def validate_benign_validators(cls, value: list[AgentConfig]) -> list[AgentConfig]:
        return _validate_agent_roles(value, "validator", "benign_validators")

    @model_validator(mode="after")
    def validate_validator_kinds(self) -> SessionConfig:
        for agent in self.attack_validators:
            if agent.config.get("kind") != "attack":
                msg = "attack_validators must have config kind 'attack'"
                raise ValueError(msg)
        for agent in self.benign_validators:
            if agent.config.get("kind") != "benign":
                msg = "benign_validators must have config kind 'benign'"
                raise ValueError(msg)
        return self

    @field_validator("storage")
    @classmethod
    def resolve_storage_root(cls, value: StorageConfig, info: ValidationInfo) -> StorageConfig:
        base_dir = info.context.get("base_dir") if info.context else None
        if base_dir is not None and not value.root_dir.is_absolute():
            return value.model_copy(update={"root_dir": Path(base_dir) / value.root_dir})
        return value

    @field_validator("preloaded_attacks")
    @classmethod
    def resolve_preloaded_attack_paths(
        cls,
        value: list[PreloadedAttackConfig],
        info: ValidationInfo,
    ) -> list[PreloadedAttackConfig]:
        base_dir = info.context.get("base_dir") if info.context else None
        if base_dir is None:
            return value
        return [
            attack.model_copy(update={"path": Path(base_dir) / attack.path})
            if not attack.path.is_absolute()
            else attack
            for attack in value
        ]

    @field_validator("benign_suite_path")
    @classmethod
    def resolve_benign_suite_path(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        base_dir = info.context.get("base_dir") if info.context else None
        if value is None or base_dir is None or value.is_absolute():
            return value
        return Path(base_dir) / value


def _validate_agent_roles(agents: list[AgentConfig], role: AgentRole, section: str) -> list[AgentConfig]:
    for agent in agents:
        if agent.role != role:
            msg = f"{section} entries must have role {role!r}"
            raise ValueError(msg)
    return agents
