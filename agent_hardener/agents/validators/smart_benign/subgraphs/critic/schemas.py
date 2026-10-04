# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schemas for the critic LLM call."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

CriticAction = Literal["keep", "drop", "rewrite"]


class CriticDecision(AgentHardenerModel):
    """One critic decision per generated-request index."""

    index: int = Field(ge=1)
    action: CriticAction
    new_payload: str | None = None
    rationale: str = Field(min_length=1)


class CriticResult(AgentHardenerModel):
    """The full critic verdict: one decision per input row, by 1-based index."""

    decisions: list[CriticDecision] = Field(default_factory=list)
