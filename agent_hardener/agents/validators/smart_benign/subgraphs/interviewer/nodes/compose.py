# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM call: generate a batch of questions for all remaining gaps in one shot."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import InterviewBatch

if TYPE_CHECKING:
    from ..state import InterviewerState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _unasked_gaps(gaps: list[str], prior_answers: list[tuple[str, str, str]]) -> list[str]:
    """Return gaps not yet covered by a prior interview round.

    Uses exact string matching on the ``gap`` field that the compose LLM copies
    verbatim into each answer tuple — no heuristic needed.
    """
    already_asked = {gap for gap, _q, _a in prior_answers if gap}
    return [g for g in gaps if g not in already_asked]


async def compose(state: InterviewerState) -> dict[str, object]:
    """Ask the LLM to generate a batch of questions, one per remaining gap."""
    if not state.gaps:
        return {"composed_questions": [], "source_note": "interviewer skipped (no gaps)"}

    unasked = _unasked_gaps(state.gaps, state.prior_answers)
    if not unasked:
        return {"composed_questions": [], "source_note": "interviewer skipped (all gaps addressed in prior Q&A)"}

    try:
        result = await _call(state, unasked)
    except RateLimitError:
        raise
    except Exception as exc:
        logger.exception("interviewer.compose LLM call failed")
        return {
            "composed_questions": [],
            "source_note": f"interviewer.compose failed ({exc}); skipping round",
            "errors": [f"interviewer.compose: {exc}"],
        }

    logger.info(
        "interviewer.compose: %d question(s) from %d gap(s)",
        len(result.questions),
        len(unasked),
    )
    return {"composed_questions": result.questions}


async def _call(state: InterviewerState, gaps: list[str]) -> InterviewBatch:
    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(InterviewBatch)
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    user_message = render_prompt(
        PROMPTS_DIR / "user_template.md",
        {
            "gaps": gaps,
            "prior_qa": state.prior_answers,
            "remaining_budget": state.remaining_budget,
            "system_role": state.system_role,
            "tool_summaries": state.tool_summaries,
            "out_of_scope": state.out_of_scope,
            "known_personas": state.known_personas,
        },
    )
    try:
        return await model.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:interviewer")
        if rate_limit is not None:
            raise rate_limit from exc
        raise
