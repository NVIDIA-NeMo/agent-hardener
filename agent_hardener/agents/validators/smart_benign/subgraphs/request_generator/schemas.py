# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the request_generator LLM call (one tool per call)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

RequestLabel = Literal["benign", "borderline_benign", "negative_control"]


class GeneratedRow(AgentHardenerModel):
    """One row the generator produces for a given tool."""

    payload: str = Field(min_length=1)
    label: RequestLabel
    rationale: str = ""
    persona: str | None = None


class ToolGenerationOutput(AgentHardenerModel):
    """The structured output of one per-tool generator call."""

    rows: list[GeneratedRow] = Field(default_factory=list)
