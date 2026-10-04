# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate benign requests per tool, in parallel.

One LLM call per tool, fanned out under ``gather_limited_ordered``. Pydantic
schema validation enforces the per-row type contract; the ``critic`` agent
downstream catches duplicates and off-topic rows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest, ToolSpec
from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import ToolGenerationOutput

if TYPE_CHECKING:
    from ..state import RequestGeneratorState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def generate(state: RequestGeneratorState) -> dict[str, object]:
    """One LLM call per tool, fanned out with bounded concurrency."""
    if not state.tools:
        return {"generated": [], "source_note": "no tools in profile; nothing to generate"}

    async def _one(tool: ToolSpec) -> tuple[list[GeneratedRequest], list[str]]:
        try:
            output = await _call(tool, state)
        except RateLimitError:
            raise
        except Exception as exc:
            logger.exception("request_generator failed for tool %r", tool.name)
            return [], [f"request_generator[{tool.name}]: {exc}"]
        return _dedupe_within_tool(_to_requests(tool, output)), []

    results = await gather_limited_ordered(state.tools, state.concurrency, _one)

    rows = [row for tool_rows, _ in results for row in tool_rows]
    errors = [err for _, tool_errs in results for err in tool_errs]
    note = f"generated {len(rows)} row(s) across {len(state.tools)} tool(s)"
    if errors:
        note += f"; {len(errors)} tool(s) failed"
    return {"generated": rows, "errors": errors, "source_note": note}


async def _call(tool: ToolSpec, state: RequestGeneratorState) -> ToolGenerationOutput:
    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(ToolGenerationOutput)
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    user_message = render_prompt(PROMPTS_DIR / "user_template.md", _prompt_vars(tool, state))
    try:
        return await model.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_message)],
        )
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source=f"smart_benign:request_generator[{tool.name}]")
        if rate_limit is not None:
            raise rate_limit from exc
        raise


def _prompt_vars(tool: ToolSpec, state: RequestGeneratorState) -> dict[str, Any]:
    return {
        "target_name": state.target_name,
        "tool": {
            "name": tool.name,
            "description": tool.description,
            "example_inputs": tool.example_inputs,
            "allowed_patterns": tool.allowed_patterns,
            "blocked_patterns": tool.blocked_patterns,
        },
        "profile": {
            "system_role": state.system_role,
            "personas": [{"name": p.name, "description": p.description} for p in state.personas],
            "out_of_scope": state.out_of_scope,
        },
        "counts": {
            "count": state.requests_per_tool,
            "borderline": state.borderline_per_tool,
            "negative_control": state.negative_control_per_tool,
        },
    }


def _dedupe_within_tool(requests: list[GeneratedRequest]) -> list[GeneratedRequest]:
    seen: set[str] = set()
    out: list[GeneratedRequest] = []
    for req in requests:
        key = req.payload.lower().split()
        key_str = " ".join(key)
        if key_str not in seen:
            seen.add(key_str)
            out.append(req)
    return out


def _to_requests(tool: ToolSpec, output: ToolGenerationOutput) -> list[GeneratedRequest]:
    return [
        GeneratedRequest(
            tool=tool.name,
            payload=row.payload,
            label=row.label,
            rationale=row.rationale,
            persona=row.persona,
        )
        for row in output.rows
    ]
