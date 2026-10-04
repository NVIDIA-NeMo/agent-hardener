# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the api_prober agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import ToolSpec
from agent_hardener.models import AgentHardenerModel


class ApiProberState(AgentHardenerModel):
    """Private state for the api_prober agent.

    Sends a bounded number of meta-prompts to the live victim endpoint via
    OpenAI chat completions, then runs one LLM extraction call to turn the
    natural-language replies into ``ToolSpec`` hypotheses tagged
    ``source="api_probe"`` with confidence capped at 0.5.
    """

    # --- Inputs (populated by entry_adapter) ---
    target_url: str
    target_name: str
    probe_model: str
    max_probes: int = Field(default=5, ge=0)
    timeout_seconds: float = Field(default=30.0, gt=0)

    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Working state (set by probe node) ---
    probe_responses: list[tuple[str, str]] = Field(default_factory=list)
    probe_errors: list[str] = Field(default_factory=list)

    # --- Output (set by extract node, read by exit_adapter) ---
    tools: list[ToolSpec] = Field(default_factory=list)
    system_role: str | None = None
    out_of_scope: list[str] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
