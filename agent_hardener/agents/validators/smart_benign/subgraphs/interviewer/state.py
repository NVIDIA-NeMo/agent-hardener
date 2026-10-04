# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the interviewer agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

from .schemas import InterviewerQuestion


class InterviewerState(AgentHardenerModel):
    """Private state for the interviewer agent.

    The LLM generates a batch of questions (one per gap, capped by
    ``remaining_budget``) in a single call. The ask node presents them to the
    user sequentially with arrow-key option pickers, then returns all collected
    answers at once for the parent to merge into ``interviewer_answers``.

    The total question budget is enforced at the parent graph level.
    """

    # --- Inputs (populated by entry_adapter) ---
    gaps: list[str] = Field(default_factory=list)
    prior_answers: list[tuple[str, str, str]] = Field(default_factory=list)
    interactive: bool = False
    remaining_budget: int = 10
    synth_model: str = ""
    synth_base_url: str = ""
    synth_api_key: str = ""
    # What is already known about the agent, so questions/options are domain-grounded.
    system_role: str | None = None
    tool_summaries: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    known_personas: list[str] = Field(default_factory=list)

    # --- Working state ---
    composed_questions: list[InterviewerQuestion] = Field(default_factory=list)

    # --- Output (read by exit_adapter) ---
    collected_answers: list[tuple[str, str, str]] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
