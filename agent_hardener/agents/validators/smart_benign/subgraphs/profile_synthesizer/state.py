# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the profile_synthesizer agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile
from agent_hardener.models import AgentHardenerModel


class ProfileSynthesizerState(AgentHardenerModel):
    """Private state for the profile_synthesizer agent.

    Merges per-source ``partials`` into a unified :class:`VictimCapabilityProfile`
    via two internal nodes: a deterministic ``merge`` (no LLM) and an LLM-backed
    ``normalize`` that fills the profile's global fields (``system_role``,
    ``personas``, ``out_of_scope``). The merge runs over tools only; interviewer
    answers feed the normalize step.
    """

    # --- Inputs (populated by entry_adapter) ---
    target_name: str
    partials: dict[str, list[ToolSpec]] = Field(default_factory=dict)
    interviewer_answers: list[tuple[str, str, str]] = Field(default_factory=list)
    source_notes: dict[str, str] = Field(default_factory=dict)
    input_hash: str
    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Working state (set by merge, read by normalize) ---
    merged_tools: list[ToolSpec] = Field(default_factory=list)

    # --- Output (read by exit_adapter) ---
    profile: VictimCapabilityProfile | None = None
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
