# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain structured-output call: turn probe transcripts into ToolSpec hypotheses."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.models import ToolSpec
from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import API_PROBE_MAX_CONFIDENCE, ProberOutput

if TYPE_CHECKING:
    from ..state import ApiProberState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def extract(state: ApiProberState) -> dict[str, object]:
    """One LLM call: extract ToolSpec hypotheses from the probe transcript."""
    if not state.probe_responses:
        return {
            "source_note": f"api_prober: no responses, {len(state.probe_errors)} error(s)",
            "errors": state.probe_errors,
        }

    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(ProberOutput)
    messages = [
        SystemMessage(content=load_prompt(PROMPTS_DIR / "system.md")),
        HumanMessage(
            content=render_prompt(
                PROMPTS_DIR / "user_template.md",
                {
                    "target_name": state.target_name,
                    "probe_qa": "\n\n".join(f"Q: {q}\nA: {a}" for q, a in state.probe_responses),
                },
            )
        ),
    ]

    try:
        output = await model.ainvoke(messages)
    except RateLimitError:
        raise
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:api_prober.extract")
        if rate_limit is not None:
            raise rate_limit from exc
        logger.exception("api_prober extract failed")
        return {
            "source_note": f"api_prober extract failed: {exc}",
            "errors": [*state.probe_errors, f"api_prober.extract: {exc}"],
        }

    tools = [
        ToolSpec(
            name=h.name,
            description=h.description,
            example_inputs=h.example_inputs,
            source="api_probe",
            confidence=min(h.confidence, API_PROBE_MAX_CONFIDENCE),
        )
        for h in output.tools
    ]
    return {
        "tools": tools,
        "system_role": output.system_role,
        "out_of_scope": output.out_of_scope,
        "source_note": f"api_prober probed {len(state.probe_responses)}/{state.max_probes}; extracted {len(tools)} tool(s)",
        "errors": list(state.probe_errors),
    }
