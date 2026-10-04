# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the profile_writer agent."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, VictimCapabilityProfile
from agent_hardener.models import AgentHardenerModel


class ProfileWriterState(AgentHardenerModel):
    """Private state for the profile_writer agent.

    Persists the unified profile, the generated request suite, per-source
    notes, and the cache key (``input_hash.txt``) to a target directory.
    Pure deterministic — no LLM, no prompts.
    """

    # --- Inputs (populated by entry_adapter) ---
    target_dir: Path
    profile: VictimCapabilityProfile | None = None
    requests: list[GeneratedRequest] = Field(default_factory=list)
    source_notes: dict[str, str] = Field(default_factory=dict)

    # --- Output (read by exit_adapter) ---
    written_files: list[Path] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
