# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain structured-output call: free-form description → typed capabilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.models import Persona, ToolSpec
from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import ParserOutput

if TYPE_CHECKING:
    from ..state import NLParserState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def parse(state: NLParserState) -> dict[str, object]:
    """One LLM call: extract the parser hypothesis from ``state.description``."""
    if not state.description:
        return {"source_note": "no description provided; nl_parser skipped"}

    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(ParserOutput)
    messages = [
        SystemMessage(content=load_prompt(PROMPTS_DIR / "system.md")),
        HumanMessage(content=render_prompt(PROMPTS_DIR / "user_template.md", {"description": state.description})),
    ]

    try:
        output: ParserOutput = await model.ainvoke(messages)
    except RateLimitError:
        raise
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:nl_parser")
        if rate_limit is not None:
            raise rate_limit from exc
        logger.exception("nl_parser failed")
        return {"errors": [f"nl_parser: {exc}"]}

    tools = [
        ToolSpec(
            name=t.name,
            description=t.description,
            example_inputs=t.example_inputs,
            source="nl",
            confidence=t.confidence,
        )
        for t in output.tools
    ]
    note = f"nl_parser found {len(tools)} tool(s)"
    if output.system_role:
        note += f"; role: {output.system_role}"
    return {
        "tools": tools,
        "personas": [Persona(name=p.name, description=p.description) for p in output.personas],
        "out_of_scope": output.out_of_scope,
        "system_role": output.system_role,
        "source_note": note,
    }
