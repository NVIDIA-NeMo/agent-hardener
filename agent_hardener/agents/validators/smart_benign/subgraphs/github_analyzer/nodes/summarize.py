# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LangChain structured-output call: README → system_role / personas / out_of_scope."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from agent_hardener.agents.validators.smart_benign.models import Persona
from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.llm import build_chat_model
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from ..schemas import SummarizerOutput

if TYPE_CHECKING:
    from ..state import GitHubAnalyzerState

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


async def summarize(state: GitHubAnalyzerState) -> dict[str, object]:
    """One LLM call over the README; merge with the YAML-derived tools.

    Tools from ``state.declared_tools`` (structural YAML parse) are passed through
    unchanged. The LLM only fills the global fields and never modifies the
    tool list.
    """
    note_parts = []
    if state.head_sha:
        note_parts.append(f"HEAD={state.head_sha[:8]}")
    note_parts.append(f"yaml_files={state.yaml_files_scanned}")
    note_parts.append(f"declared_tools={len(state.declared_tools)}")

    if not state.readme_text:
        note_parts.append("readme=absent")
        return {"tools": state.declared_tools, "source_note": "github_analyzer: " + ", ".join(note_parts)}

    note_parts.append(f"readme={len(state.readme_text)}b")

    model = build_chat_model(
        model=state.synth_model,
        base_url=state.synth_base_url,
        api_key=state.synth_api_key,
    ).with_structured_output(SummarizerOutput)
    tool_summary = "\n".join(f"- {t.name}: {t.description}" for t in state.declared_tools) or "(none extracted)"
    messages = [
        SystemMessage(content=load_prompt(PROMPTS_DIR / "system.md")),
        HumanMessage(
            content=render_prompt(
                PROMPTS_DIR / "user_template.md",
                {
                    "target_name": state.target_name,
                    "tool_summary": tool_summary,
                    "readme": state.readme_text,
                },
            )
        ),
    ]

    try:
        output: SummarizerOutput = await model.ainvoke(messages)
    except RateLimitError:
        raise
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source="smart_benign:github_analyzer.summarize")
        if rate_limit is not None:
            raise rate_limit from exc
        logger.exception("github_analyzer summarize failed")
        note_parts.append(f"summarize_failed={exc.__class__.__name__}")
        return {
            "tools": state.declared_tools,
            "source_note": "github_analyzer: " + ", ".join(note_parts),
            "errors": [f"github_analyzer.summarize: {exc}"],
        }

    note_parts.append(f"personas={len(output.personas)}")
    return {
        "tools": state.declared_tools,
        "personas": [Persona(name=p.name, description=p.description) for p in output.personas],
        "system_role": output.system_role,
        "out_of_scope": output.out_of_scope,
        "source_note": "github_analyzer: " + ", ".join(note_parts),
    }
