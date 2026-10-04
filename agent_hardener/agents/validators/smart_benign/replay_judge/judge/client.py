# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refusal judge: one LangChain structured-output call."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import rate_limit_error_from_exception

from .schema import JudgeVerdict

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


async def judge_one(
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    tool: str,
    payload: str,
    response: str,
    max_tokens: int,
    timeout: float,
    temperature: float | None = None,
) -> JudgeVerdict:
    """Invoke the judge for one response and return a typed verdict.

    ``max_tokens`` caps the verdict generation and ``timeout`` bounds the request so a slow judge
    can't stall a row indefinitely. Schema validation is enforced by ``with_structured_output``; if
    the model emits malformed JSON or violates the schema, LangChain raises and the caller
    (``runner._process_one``) records an error verdict for that row.
    """
    model = build_chat_model(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_tokens,
        timeout=timeout,
        temperature=temperature,
    ).with_structured_output(JudgeVerdict)
    system_prompt = load_prompt(PROMPTS_DIR / "system.md")
    user_message = render_prompt(
        PROMPTS_DIR / "user_template.md",
        {"tool": tool, "payload": payload, "response": response},
    )
    try:
        result = await model.ainvoke([SystemMessage(content=system_prompt), HumanMessage(content=user_message)])
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:judge")
        if rate_limit is not None:
            raise rate_limit from exc
        raise
    return result
