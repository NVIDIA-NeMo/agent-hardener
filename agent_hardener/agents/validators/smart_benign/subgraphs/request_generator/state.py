# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the request_generator agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, Persona, ToolSpec
from agent_hardener.models import AgentHardenerModel


class RequestGeneratorState(AgentHardenerModel):
    """Private state for the request_generator agent.

    Produces up to N benign rows, M borderline-benign rows, and K
    negative-control rows per tool. Per-tool LLM calls run in parallel
    under ``gather_limited_ordered``.
    """

    # --- Inputs (populated by entry_adapter) ---
    target_name: str
    tools: list[ToolSpec] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    system_role: str | None = None
    out_of_scope: list[str] = Field(default_factory=list)

    requests_per_tool: int = Field(default=3, ge=1)
    borderline_per_tool: int = Field(default=2, ge=0)
    negative_control_per_tool: int = Field(default=1, ge=0)
    concurrency: int = Field(default=2, ge=1)

    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Output (read by exit_adapter) ---
    generated: list[GeneratedRequest] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
