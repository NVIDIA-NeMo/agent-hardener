# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the interviewer agent's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import InterviewerState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def entry_adapter(parent: SynthState) -> InterviewerState:
    """Pull gaps, conversation history, the known agent context, and LLM config."""
    d = defaults()
    profile = parent.profile
    tools = profile.tools if profile else []
    return InterviewerState(
        gaps=list(parent.gaps),
        prior_answers=list(parent.interviewer_answers),
        interactive=parent.interactive,
        remaining_budget=parent.max_interview_questions - parent.interview_questions_asked,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
        system_role=profile.system_role if profile else None,
        tool_summaries=[f"{t.name} — {t.description}" for t in tools],
        out_of_scope=list(profile.out_of_scope) if profile else [],
        known_personas=[f"{p.name}: {p.description}" for p in profile.personas] if profile else [],
    )


def exit_adapter(sub: InterviewerState) -> dict[str, object]:
    """Surface the composed questions (serialized) for the parent's ask node to interrupt on."""
    delta: dict[str, object] = {
        "composed_questions": [
            {
                "gap": q.gap,
                "question": q.question,
                "options": [
                    {"label": o.label, "description": o.description, "recommended": o.recommended} for o in q.options
                ],
            }
            for q in sub.composed_questions
            if q.question is not None
        ]
    }
    if sub.source_note:
        delta["source_notes"] = {"interviewer": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
