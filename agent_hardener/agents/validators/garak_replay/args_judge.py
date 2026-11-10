# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Did a failed call carry out the attack? The judge behind ``tool_failed`` hits.

A ``tool_failed`` hit is one the detector judged a success whose target tool was called and errored,
with nothing of ours blocking it. Whether that attack landed depends on the call's arguments: a call
that carried the attacker's target or action is a landed attack even though it failed; an ordinary
call that ignored the attack is not. One structured-output call per such hit decides.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from agent_hardener.agents.validators.smart_benign.prompt_loader import load_prompt, render_prompt
from agent_hardener.env import inference_api_key
from agent_hardener.llm import build_chat_model
from agent_hardener.models import AgentHardenerModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_hardener.models import AgentConfig

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
#: Per-field clip for what the judge is shown.
_CLIP = 2000


class ArgsVerdict(AgentHardenerModel):
    """The judge's whole answer."""

    carries_attack: bool
    reasoning: str = Field(min_length=3)


def _arguments_text(arguments: Any) -> str:
    return json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)[:_CLIP]


def build_args_judge(agent: AgentConfig) -> Callable[[str, str, Any], Awaitable[bool]] | None:
    """The judge for ``tool_failed`` hits, or ``None`` without an API key (those hits then count as landed).

    Configured from ``config.args_judge`` (``model``, ``base_url``, ``api_key``, ``max_tokens``,
    ``timeout_seconds``), falling back to the smart benign judge defaults, as the refusal judge does.
    """
    from agent_hardener.agents.validators.smart_benign.config import defaults  # noqa: PLC0415

    raw: dict[str, Any] = agent.config.get("args_judge") or {}
    api_key = str(raw.get("api_key") or inference_api_key() or "")
    if not api_key:
        return None
    d = defaults()

    async def judge(tool: str, prompt: str, arguments: Any) -> bool:
        model = build_chat_model(
            model=str(raw.get("model") or d.judge_llm.model),
            base_url=str(raw.get("base_url") or d.judge_llm.base_url),
            api_key=api_key,
            max_tokens=int(raw.get("max_tokens") or d.judge_llm.max_tokens),
            timeout=float(raw.get("timeout_seconds") or d.judge_llm.timeout_seconds),
            temperature=d.judge_llm.temperature,
        ).with_structured_output(ArgsVerdict)
        variables = {"prompt": prompt[:_CLIP], "tool": tool, "arguments": _arguments_text(arguments)}
        messages = [
            SystemMessage(content=load_prompt(PROMPTS_DIR / "args_judge.system.md")),
            HumanMessage(content=render_prompt(PROMPTS_DIR / "args_judge.user.md", variables)),
        ]
        verdict = await model.ainvoke(messages)
        return bool(ArgsVerdict.model_validate(verdict).carries_attack)

    return judge
