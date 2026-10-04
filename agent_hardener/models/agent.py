# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-wrapper configuration and the typed target passed to every agent."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import AliasChoices, Field, PrivateAttr, field_validator, model_validator

from agent_hardener.ids import build_agent_id
from agent_hardener.models.base import AgentHardenerModel, AgentRole

if TYPE_CHECKING:
    from collections.abc import Mapping


class TargetInput(AgentHardenerModel):
    """Typed target input passed to every agent wrapper."""

    name: str = Field(min_length=1)
    base_url: str | None = None
    #: Host path to the run's NeMo Relay ``plugins.toml`` — the guardrail set the defenders extend
    #: and the deploy stage uploads into the victim.
    agent_relay_plugins: Path | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentConfig(AgentHardenerModel):
    """Configuration for one external agent wrapper."""

    name: str = Field(min_length=1)
    role: AgentRole
    service_url: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0)
    implementation: str | None = Field(
        default=None,
        validation_alias=AliasChoices("implementation", "external_implementation"),
    )
    config: dict[str, Any] = Field(default_factory=dict)
    capabilities: str | None = Field(
        default=None,
        description="Natural language description of what attacks this defender mitigates.",
    )
    _agent_id_cache: str = PrivateAttr(default="")

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "agent name must not be blank"
            raise ValueError(msg)
        return stripped

    @model_validator(mode="after")
    def validate_role_constraints(self) -> AgentConfig:
        if self.role == "validator":
            kind = self.config.get("kind")
            if kind is not None and kind not in {"attack", "benign"}:
                msg = "validator config kind must be 'attack' or 'benign'"
                raise ValueError(msg)
        if self.role == "defender" and not self.capabilities:
            msg = "defender agents must have a non-empty 'capabilities' field"
            raise ValueError(msg)
        return self

    def model_post_init(self, _context: Any) -> None:
        self._agent_id_cache = build_agent_id(self)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> AgentConfig:
        copy = super().model_copy(update=update, deep=deep)
        copy._agent_id_cache = build_agent_id(copy)
        return copy

    @property
    def agent_id(self) -> str:
        return self._agent_id_cache
