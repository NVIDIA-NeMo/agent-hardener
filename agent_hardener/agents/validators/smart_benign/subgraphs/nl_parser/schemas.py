# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schemas for the nl_parser LLM call.

These are the shapes returned by ``ChatNVIDIA(...).with_structured_output(ParserOutput)``
inside ``nodes/parse.py``. The ``ParserToolHypothesis.name`` regex enforces
snake_case at schema-validation time so the LLM cannot produce drift.
"""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel


class ParserToolHypothesis(AgentHardenerModel):
    """One tool the LLM thinks the described system exposes."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    example_inputs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ParserPersona(AgentHardenerModel):
    """One user persona the LLM infers from the description."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ParserOutput(AgentHardenerModel):
    """The full structured response returned by the parser LLM call."""

    system_role: str | None = None
    tools: list[ParserToolHypothesis] = Field(default_factory=list)
    personas: list[ParserPersona] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
