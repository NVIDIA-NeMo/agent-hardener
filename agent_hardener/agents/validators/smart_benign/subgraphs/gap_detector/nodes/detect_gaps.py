# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect gaps in a merged VictimCapabilityProfile using deterministic rules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile

    from ..state import GapDetectorState

logger = logging.getLogger(__name__)

COMPLETENESS_GAP = "tools completeness: confirm no additional tools are missing from the profile"


async def detect_gaps(state: GapDetectorState) -> dict[str, object]:
    """Apply gap-detection rules and return the resulting list."""
    if state.profile is None:
        return {"gaps": ["profile is missing — no ingestion subgraph produced any output"]}

    gaps: list[str] = []
    gaps.extend(_tool_gaps(state.profile.tools, state.confidence_threshold))
    gaps.extend(_global_gaps(state.profile))
    gaps.extend(_dropped_tool_gaps(state.profile.tools, state.partials))
    gaps.extend(_merge_proposal_gaps(state.profile))
    gaps.extend(_completeness_gaps(state.profile))
    logger.info("gap_detector found %d gap(s)", len(gaps))
    return {"gaps": gaps}


def _tool_gaps(tools: list[ToolSpec], confidence_threshold: float) -> list[str]:
    if not tools:
        return ["no tools were inferred for the victim"]
    out: list[str] = []
    for tool in tools:
        if not tool.description.strip():
            out.append(f"tool {tool.name!r} has no description")
        if not tool.example_inputs:
            out.append(f"tool {tool.name!r} has no example_inputs")
        if tool.confidence < confidence_threshold:
            out.append(f"tool {tool.name!r} has low confidence ({tool.confidence:.2f} < {confidence_threshold:.2f})")
    return out


def _global_gaps(profile: VictimCapabilityProfile) -> list[str]:
    out: list[str] = []
    if not profile.system_role:
        out.append("system_role is missing")
    if not profile.personas:
        out.append("no personas defined")
    if not profile.out_of_scope:
        out.append("out_of_scope is not declared")
    return out


def _dropped_tool_gaps(profile_tools: list[ToolSpec], partials: dict[str, list[ToolSpec]]) -> list[str]:
    """Flag tools found by an ingestion source but absent from the final profile."""
    if not partials:
        return []
    profile_names = {t.name for t in profile_tools}
    out: list[str] = []
    for source, tools in partials.items():
        for tool in tools:
            if tool.name not in profile_names:
                out.append(f"tool {tool.name!r} was found by source {source!r} but is absent from the final profile")
    return out


def _merge_proposal_gaps(profile: VictimCapabilityProfile) -> list[str]:
    """Emit one confirmation gap per LLM-proposed tool-name merge."""
    out: list[str] = []
    for proposal in profile.proposed_tool_merges:
        aliases_str = ", ".join(f"'{a}'" for a in proposal.aliases)
        out.append(
            f"tool merge proposal: '{proposal.canonical_name}' and {aliases_str} "
            f"may be the same tool — confirm and choose canonical name"
        )
    return out


def _completeness_gaps(profile: VictimCapabilityProfile) -> list[str]:
    """Emit a recall prompt when at least one tool has been inferred.

    Stable gap string so the interviewer's prior-answer deduplication suppresses
    re-asking once the user has confirmed the tool list is complete.
    """
    if not profile.tools:
        return []
    return [COMPLETENESS_GAP]
