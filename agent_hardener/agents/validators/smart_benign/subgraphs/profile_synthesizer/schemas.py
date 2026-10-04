# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schemas for the profile_synthesizer LLM call."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel


class NormalizedPersona(AgentHardenerModel):
    """One user persona inferred by the normalize step."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class ToolPatch(AgentHardenerModel):
    """LLM-supplied corrections for a single tool's fields.

    Populated when interview answers fill gaps the merge step left blank, or
    when a ``boundary[...]`` interview answer clarifies the tool's benign/attack
    line (``allowed_patterns`` / ``blocked_patterns``). Set ``drop=True`` to
    remove the tool from the profile entirely (used when a merge approval drops
    an alias in favour of the canonical name).
    """

    name: str = Field(min_length=1)
    description: str | None = None
    example_inputs: list[str] = Field(default_factory=list)
    allowed_patterns: list[str] = Field(default_factory=list)
    blocked_patterns: list[str] = Field(default_factory=list)
    drop: bool = False


class ToolMergeProposal(AgentHardenerModel):
    """LLM hypothesis that two or more tool names refer to the same capability.

    Emitted by the normalize step when it notices names that look like aliases
    (e.g. ``shell_executor`` vs ``bash_executor``). The gap detector converts
    each proposal into a confirmation gap; once the user approves, the next
    normalize pass executes the merge via ``ToolPatch``.
    """

    canonical_name: str = Field(min_length=1)
    aliases: list[str] = Field(min_length=1)
    rationale: str = Field(min_length=1)


class NormalizedProfileGlobals(AgentHardenerModel):
    """LLM-produced global fields for the unified profile.

    The tool list is built deterministically by the ``merge`` node; the LLM
    fills the three global slots, patches tool-level gaps from interview
    answers, and proposes tool-name merges for user confirmation.
    """

    system_role: str | None = None
    personas: list[NormalizedPersona] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    tool_patches: list[ToolPatch] = Field(default_factory=list)
    proposed_tool_merges: list[ToolMergeProposal] = Field(default_factory=list)
