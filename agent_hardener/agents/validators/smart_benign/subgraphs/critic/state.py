# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the critic agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.models import AgentHardenerModel


class CriticState(AgentHardenerModel):
    """Private state for the critic agent.

    Reviews the generated benign suite for duplicates, off-topic rows, and
    rows that contradict declared out_of_scope rules. Outputs a filtered (and
    optionally rewritten) request list. Cannot trigger regeneration — its
    authority is bounded to drop/rewrite.
    """

    # --- Inputs (populated by entry_adapter) ---
    target_name: str
    requests: list[GeneratedRequest] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Output (read by exit_adapter) ---
    filtered_requests: list[GeneratedRequest] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
