# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic merge of per-source partial ToolSpec lists."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.models import ToolSource, ToolSpec

    from ..state import ProfileSynthesizerState

logger = logging.getLogger(__name__)

SOURCE_PRIORITY: Final[dict[ToolSource, int]] = {
    "github": 3,
    "nl": 2,
    "api_probe": 1,
    "interview": 0,
}
"""Tool-merge precedence. Higher beats lower when the same tool appears in
multiple sources. Structural sources (``github``) outrank inferential ones
(``nl``, ``api_probe``); ``interview`` is the lowest because the interviewer
is meant to clarify global fields, not add tools."""


async def merge(state: ProfileSynthesizerState) -> dict[str, object]:
    """Dedupe by tool name, pick the highest-priority source, union examples."""
    if not state.partials:
        logger.info("profile_synthesizer.merge: no partials")
        return {"merged_tools": []}

    by_name: dict[str, list[ToolSpec]] = {}
    for tools in state.partials.values():
        for tool in tools:
            by_name.setdefault(tool.name, []).append(tool)

    merged: list[ToolSpec] = []
    for candidates in by_name.values():
        winner = max(candidates, key=_priority_key)
        unioned_examples = _union_examples(candidates)
        merged.append(winner.model_copy(update={"example_inputs": unioned_examples}))

    logger.info(
        "profile_synthesizer.merge produced %d unique tool(s) from %d source(s)", len(merged), len(state.partials)
    )
    return {"merged_tools": merged}


def _priority_key(tool: ToolSpec) -> int:
    return SOURCE_PRIORITY.get(tool.source, -1)


def _union_examples(candidates: list[ToolSpec]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for candidate in candidates:
        for example in candidate.example_inputs:
            if example not in seen:
                seen.add(example)
                out.append(example)
    return out
