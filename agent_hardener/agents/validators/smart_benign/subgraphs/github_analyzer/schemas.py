# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the github_analyzer summarize LLM call."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel


class SummarizerPersona(AgentHardenerModel):
    """One user persona inferred from the README."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class SummarizerOutput(AgentHardenerModel):
    """LLM-derived globals extracted from a repo's README.

    The tool list is built deterministically by the YAML scan, not by this
    LLM step — the schema deliberately does not include a ``tools`` field
    so the model can't override the structural tool extraction.
    """

    system_role: str | None = None
    personas: list[SummarizerPersona] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
