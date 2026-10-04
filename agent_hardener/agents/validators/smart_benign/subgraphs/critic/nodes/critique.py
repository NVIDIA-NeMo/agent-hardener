# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain structured-output call: critic decisions + deterministic dedup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import CriticDecision, CriticResult

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest

    from ..state import CriticState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def critique(state: CriticState) -> dict[str, object]:
    """Review every request, then dedupe deterministically as a safety net."""
    if not state.requests:
        return {"filtered_requests": [], "source_note": "no requests to critique"}

    try:
        result = await _call_critic(state)
    except RateLimitError:
        raise
    except Exception as exc:
        logger.exception("critic failed")
        deduped = _dedupe_exact(state.requests)
        return {
            "filtered_requests": deduped,
            "source_note": f"critic LLM failed ({exc}); deterministic dedup only: {len(state.requests)} → {len(deduped)}",
            "errors": [f"critic: {exc}"],
        }

    after_llm = _apply_decisions(state.requests, result.decisions)
    after_dedup = _dedupe_exact(after_llm)
    return {
        "filtered_requests": after_dedup,
        "source_note": (
            f"critic kept {len(after_llm)}/{len(state.requests)} (LLM); {len(after_dedup)} after exact dedup"
        ),
    }


async def _call_critic(state: CriticState) -> CriticResult:
    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(CriticResult)
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    user_message = render_prompt(
        PROMPTS_DIR / "user_template.md",
        {
            "target_name": state.target_name,
            "tools_summary": _format_tool_names(state.tool_names),
            "out_of_scope": _format_list(state.out_of_scope),
            "requests_table": _format_requests(state.requests),
        },
    )
    try:
        return await model.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:critic")
        if rate_limit is not None:
            raise rate_limit from exc
        raise


def _apply_decisions(
    requests: list[GeneratedRequest],
    decisions: list[CriticDecision],
) -> list[GeneratedRequest]:
    by_index: dict[int, CriticDecision] = {}
    for decision in decisions:
        if decision.index in by_index:
            logger.warning("critic emitted multiple decisions for index %d; last wins", decision.index)
        by_index[decision.index] = decision

    out: list[GeneratedRequest] = []
    for one_based, request in enumerate(requests, start=1):
        decision = by_index.get(one_based)
        if decision is None:
            logger.warning("critic missed index %d; keeping by default", one_based)
            out.append(request)
            continue
        if decision.action == "drop":
            continue
        if decision.action == "rewrite":
            if decision.new_payload:
                out.append(request.model_copy(update={"payload": decision.new_payload}))
            else:
                logger.warning("critic chose 'rewrite' at index %d without new_payload; keeping original", one_based)
                out.append(request)
            continue
        # keep
        out.append(request)
    return out


def _dedupe_exact(requests: list[GeneratedRequest]) -> list[GeneratedRequest]:
    seen: set[tuple[str, str]] = set()
    out: list[GeneratedRequest] = []
    for request in requests:
        key = (request.tool, request.payload)
        if key in seen:
            continue
        seen.add(key)
        out.append(request)
    return out


def _format_requests(requests: list[GeneratedRequest]) -> str:
    return "\n".join(
        f"[{i}] tool={r.tool} label={r.label} payload={r.payload!r}" for i, r in enumerate(requests, start=1)
    )


def _format_tool_names(names: list[str]) -> str:
    if not names:
        return "(none)"
    return "\n".join(f"- {n}" for n in names)


def _format_list(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)
