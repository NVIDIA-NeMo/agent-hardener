# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Domain models for the smart benign validator.

These are the typed artifacts produced and consumed across the synth DAG:

- :class:`ToolSpec` — one entry describing a victim-exposed tool
- :class:`Persona` — a user persona used to diversify generated requests
- :class:`VictimCapabilityProfile` — the central profile the DAG builds
- :class:`GeneratedRequest` — one row in the generated benign suite

The transient pipeline state lives in :mod:`state` (kept separate so the
domain artifacts here can be reused by tests, the CLI, and any post-hoc
analysis without pulling in LangGraph types).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator

from agent_hardener.models import AgentHardenerModel

ToolSource = Literal["nl", "github", "api_probe", "interview"]
"""Where in the synth DAG a tool-spec hypothesis originated."""

RequestLabel = Literal["benign", "borderline_benign", "negative_control"]
"""Classification of a generated request.

- ``benign``: ordinary legitimate use of the tool
- ``borderline_benign``: brushes against a blocked pattern but should be allowed
- ``negative_control``: clearly out-of-scope; expected refusal validates the judge
"""


class ToolSpec(AgentHardenerModel):
    """One tool the victim is believed to expose."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters_schema: dict[str, Any] | None = None
    allowed_patterns: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)
    example_inputs: list[str] = Field(default_factory=list)
    source: ToolSource
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "tool name must not be blank"
            raise ValueError(msg)
        return stripped


class Persona(AgentHardenerModel):
    """A user persona used to diversify generated request phrasing."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ToolMergeProposal(AgentHardenerModel):
    """Hypothesis that two or more tool names refer to the same capability.

    Carried on the profile so the gap detector can emit confirmation gaps
    without needing a separate channel from the profile synthesizer.
    """

    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class VictimCapabilityProfile(AgentHardenerModel):
    """Unified description of a victim's legitimate capabilities.

    Produced by the synth DAG (merged from NL / GitHub / API-probe / interview
    sources); consumed by the request generator and persisted to the cache.
    """

    target_name: str = Field(min_length=1)
    system_role: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    proposed_tool_merges: list[ToolMergeProposal] = Field(default_factory=list)
    source_notes: dict[str, str] = Field(default_factory=dict)
    input_hash: str = Field(min_length=1)
    generated_at: datetime


class GeneratedRequest(AgentHardenerModel):
    """One row in the generated benign suite that will be replayed."""

    tool: str = Field(min_length=1)
    payload: str = Field(min_length=1)
    label: RequestLabel
    rationale: str = ""
    persona: str | None = None
