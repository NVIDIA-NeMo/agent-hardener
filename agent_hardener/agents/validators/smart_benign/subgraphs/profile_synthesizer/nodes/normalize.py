# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain structured-output call: normalize the profile's global fields."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.models import (
    Persona,
    ToolMergeProposal,
    ToolSpec,
    VictimCapabilityProfile,
)
from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import NormalizedProfileGlobals, ToolPatch

if TYPE_CHECKING:
    from ..state import ProfileSynthesizerState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def normalize(state: ProfileSynthesizerState) -> dict[str, object]:
    """Build the unified profile: merged tools + LLM-normalized globals."""
    now = datetime.now(tz=UTC)

    if not state.merged_tools:
        return {
            "profile": VictimCapabilityProfile(
                target_name=state.target_name,
                tools=[],
                personas=[],
                out_of_scope=[],
                source_notes=state.source_notes,
                input_hash=state.input_hash,
                generated_at=now,
            ),
            "source_note": "no tools discovered",
        }

    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(NormalizedProfileGlobals)
    tool_summary = "\n".join(
        f"- {t.name} ({t.source}, confidence={t.confidence:.2f}): {t.description}" for t in state.merged_tools[:20]
    )
    if len(state.merged_tools) > 20:
        tool_summary += f"\n... and {len(state.merged_tools) - 20} more"
    messages = [
        SystemMessage(content=load_prompt(PROMPTS_DIR / "system.md")),
        HumanMessage(
            content=render_prompt(
                PROMPTS_DIR / "user_template.md",
                {
                    "target_name": state.target_name,
                    "tool_summary": tool_summary,
                    "tool_gaps": _format_tool_gaps(state.merged_tools),
                    "source_notes": "\n".join(f"- {s}: {n}" for s, n in state.source_notes.items()) or "(none)",
                    "interview_qa": "\n".join(
                        (f"Gap: {g}\n" if g else "") + f"Q: {q}\nA: {a}" for g, q, a in state.interviewer_answers
                    )
                    or "(none)",
                },
            )
        ),
    ]

    try:
        output: NormalizedProfileGlobals = await model.ainvoke(messages)
    except RateLimitError:
        raise
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:profile_synthesizer")
        if rate_limit is not None:
            raise rate_limit from exc
        logger.exception("profile_synthesizer.normalize failed")
        return {
            "profile": VictimCapabilityProfile(
                target_name=state.target_name,
                tools=state.merged_tools,
                personas=[],
                out_of_scope=[],
                source_notes=state.source_notes,
                input_hash=state.input_hash,
                generated_at=now,
            ),
            "source_note": f"profile (LLM normalize failed, globals empty): {exc}",
            "errors": [f"profile_synthesizer.normalize: {exc}"],
        }

    personas = [Persona(name=p.name, description=p.description) for p in output.personas]
    patched_tools = _apply_tool_patches(state.merged_tools, output.tool_patches)
    merge_proposals = [
        ToolMergeProposal(
            canonical_name=m.canonical_name,
            aliases=m.aliases,
            rationale=m.rationale,
        )
        for m in output.proposed_tool_merges
    ]
    return {
        "profile": VictimCapabilityProfile(
            target_name=state.target_name,
            system_role=output.system_role,
            tools=patched_tools,
            personas=personas,
            out_of_scope=output.out_of_scope,
            proposed_tool_merges=merge_proposals,
            source_notes=state.source_notes,
            input_hash=state.input_hash,
            generated_at=now,
        ),
        "source_note": f"merged {len(patched_tools)} tool(s); {len(personas)} persona(s)",
    }


def _format_tool_gaps(tools: list[ToolSpec]) -> str:
    lines = []
    for t in tools:
        if not t.description.strip():
            lines.append(f"- {t.name}: missing description")
        if not t.example_inputs:
            lines.append(f"- {t.name}: missing example_inputs")
    return "\n".join(lines) if lines else "(none)"


def _apply_tool_patches(tools: list[ToolSpec], patches: list[ToolPatch]) -> list[ToolSpec]:
    if not patches:
        return tools
    known_names = {t.name for t in tools}
    by_name = {p.name: p for p in patches}
    result = []
    for tool in tools:
        patch = by_name.get(tool.name)
        if patch is None:
            result.append(tool)
            continue
        if patch.drop:
            continue  # merge approval dropped this alias
        updates: dict[str, object] = {}
        if patch.description and not tool.description.strip():
            updates["description"] = patch.description
        if patch.example_inputs and not tool.example_inputs:
            updates["example_inputs"] = patch.example_inputs
        # Boundary clarifications refine the line, so union rather than fill-if-empty.
        if patch.allowed_patterns:
            updates["allowed_patterns"] = list(dict.fromkeys([*tool.allowed_patterns, *patch.allowed_patterns]))
        if patch.blocked_patterns:
            updates["blocked_patterns"] = list(dict.fromkeys([*tool.blocked_patterns, *patch.blocked_patterns]))
        result.append(tool.model_copy(update=updates) if updates else tool)
    for patch in patches:
        if patch.name not in known_names and not patch.drop and patch.description:
            result.append(
                ToolSpec(
                    name=patch.name,
                    description=patch.description,
                    example_inputs=patch.example_inputs,
                    allowed_patterns=patch.allowed_patterns,
                    blocked_patterns=patch.blocked_patterns,
                    source="interview",
                    confidence=0.7,
                )
            )
    return result
