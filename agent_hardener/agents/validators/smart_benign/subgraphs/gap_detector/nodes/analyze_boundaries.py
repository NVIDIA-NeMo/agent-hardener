# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM boundary pass: surface where each tool's benign/attack line is unclear.

Runs after the deterministic ``detect_gaps`` rules, and only while the interview
is still open (``interview_open`` — interactive and the round/question budget is
not yet spent); otherwise there is no way to ask the questions, so the fan-out is
skipped entirely. One LLM call per tool, fanned out under ``gather_limited_ordered``. Each emitted gap is prefixed with the stable
per-tool key ``boundary[<tool>]:`` so a later round can tell which tools the
user has already clarified — this is what keeps the interview loop from
re-asking when the LLM phrases the question differently each round.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import BoundaryVerdict

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.models import ToolSpec

    from ..state import GapDetectorState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

_BOUNDARY_PREFIX = "boundary["


def boundary_key(tool_name: str) -> str:
    """Return the stable per-tool gap prefix used for dedup across rounds."""
    return f"{_BOUNDARY_PREFIX}{tool_name}]:"


def _answered_boundary_tools(prior_answers: list[tuple[str, str, str]]) -> set[str]:
    """Tool names whose ``boundary[...]`` gap already has an interview answer."""
    answered: set[str] = set()
    for gap, _q, _a in prior_answers:
        if gap.startswith(_BOUNDARY_PREFIX):
            answered.add(gap[len(_BOUNDARY_PREFIX) :].split("]", 1)[0])
    return answered


async def analyze_boundaries(state: GapDetectorState) -> dict[str, object]:
    """Append ``boundary[<tool>]:`` gaps for tools whose line is unclear."""
    if not state.interview_open or state.profile is None or not state.profile.tools:
        return {}

    answered = _answered_boundary_tools(state.interviewer_answers)
    pending = [t for t in state.profile.tools if t.name not in answered]
    if not pending:
        return {}

    async def _one(tool: ToolSpec) -> str | None:
        try:
            verdict = await _call(tool, state)
        except RateLimitError:
            raise
        except Exception:
            logger.exception("analyze_boundaries failed for tool %r", tool.name)
            return None
        if verdict.needs_clarification and verdict.question.strip():
            return f"{boundary_key(tool.name)} {verdict.question.strip()}"
        return None

    results = await gather_limited_ordered(pending, state.concurrency, _one)
    boundary_gaps = [gap for gap in results if gap]
    logger.info("analyze_boundaries added %d boundary gap(s)", len(boundary_gaps))
    return {"gaps": [*state.gaps, *boundary_gaps]}


async def _call(tool: ToolSpec, state: GapDetectorState) -> BoundaryVerdict:
    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(BoundaryVerdict)
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    user_message = render_prompt(PROMPTS_DIR / "user_template.md", _prompt_vars(tool, state))
    try:
        return await model.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source=f"smart_benign:gap_detector[{tool.name}]")
        if rate_limit is not None:
            raise rate_limit from exc
        raise


def _prompt_vars(tool: ToolSpec, state: GapDetectorState) -> dict[str, Any]:
    profile = state.profile
    return {
        "target_name": profile.target_name if profile else "",
        "tool": {
            "name": tool.name,
            "description": tool.description,
            "example_inputs": tool.example_inputs,
            "allowed_patterns": tool.allowed_patterns,
            "blocked_patterns": tool.blocked_patterns,
        },
        "out_of_scope": list(profile.out_of_scope) if profile else [],
    }
