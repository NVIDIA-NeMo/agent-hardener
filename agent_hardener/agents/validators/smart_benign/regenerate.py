# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benign rows for tools the victim's traces show but the synthesized profile does not list.

:mod:`.verify` collects those tools with their observed calls. Each becomes a :class:`ToolSpec`
(``source="trace"``) in two steps: :func:`extract_spec` builds it from the calls alone (deterministic), then
:func:`refine_spec` asks the synth LLM for a cleaner description and extra example requests (best effort).
Then only the ``request_generator`` -> ``critic`` part of the synth DAG runs for them, on a profile holding
just the new tools: the rest of the suite is not touched.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import Field

from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.env import inference_api_key
from agent_hardener.llm import build_chat_model
from agent_hardener.models.base import AgentHardenerModel
from agent_hardener.rate_limits import RateLimitError, rate_limit_error_from_exception

from .config import defaults
from .models import ToolSpec
from .prompt_loader import load_prompt, render_prompt
from .subgraphs import critic, request_generator

if TYPE_CHECKING:
    from collections.abc import Callable

    from .models import GeneratedRequest
    from .state import SynthState
    from .verify import ObservedCall

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"
_ARGS_CHARS = 500
_DESCRIPTION_ARGS_CHARS = 200
_MAX_EXTRA_EXAMPLES = 3


class ToolRefinement(AgentHardenerModel):
    """The LLM's refinement of an extracted tool spec."""

    description: str = Field(min_length=1)
    extra_example_inputs: list[str] = Field(default_factory=list)


async def describe_observed_tools(
    observed: dict[str, list[ObservedCall]], system_role: str | None
) -> tuple[list[ToolSpec], list[str]]:
    """One :class:`ToolSpec` per observed tool, and the errors of any refinement that failed.

    A failed refinement keeps the extracted spec: the victim was seen calling the tool, so it exists whatever
    the LLM did.
    """
    settings = defaults().synth_llm

    async def one(item: tuple[str, list[ObservedCall]]) -> tuple[ToolSpec, str | None]:
        name, calls = item
        spec = extract_spec(name, calls)
        try:
            return await refine_spec(spec, calls, system_role), None
        except RateLimitError:
            raise
        except Exception as exc:
            logger.info("refining observed tool %r failed; keeping the extracted spec", name, exc_info=True)
            return spec, f"observed_tool[{name}]: {exc}"

    results = cast(
        "list[tuple[ToolSpec, str | None]]",
        await gather_limited_ordered(list(observed.items()), settings.concurrency, one),
    )
    return [spec for spec, _ in results], [err for _, err in results if err]


def extract_spec(name: str, calls: list[ObservedCall]) -> ToolSpec:
    """The spec the trace evidence alone supports: a name, the arguments seen, and the prompts that called it.

    The victim was seen calling it, so it exists: confidence 1 and ``verified``. The triggering prompts are
    the strongest examples there are, since each one is known to make the agent call the tool.
    """
    args = next((call.args for call in calls if call.args not in (None, {}, [], "")), None)
    seen = f" with arguments like {_clip(json.dumps(args, default=str), _DESCRIPTION_ARGS_CHARS)}" if args else ""
    return ToolSpec(
        name=name,
        description=f"Tool `{name}` the agent was observed calling{seen}.",
        example_inputs=list(dict.fromkeys(call.prompt for call in calls)),
        source="trace",
        confidence=1.0,
        verification="verified",
    )


async def refine_spec(spec: ToolSpec, calls: list[ObservedCall], system_role: str | None) -> ToolSpec:
    """``spec`` with the LLM's cleaner description, and its extra example requests after the observed ones."""
    settings = defaults().synth_llm
    model = build_chat_model(
        model=settings.model,
        base_url=settings.base_url,
        api_key=inference_api_key() or "",
        max_tokens=settings.max_tokens,
    ).with_structured_output(ToolRefinement)
    variables = {
        "tool": spec.name,
        "system_role": system_role or "(unknown)",
        "description": spec.description,
        "calls": _render_calls(calls),
    }
    messages = [
        SystemMessage(content=load_prompt(PROMPTS_DIR / "observed_tool.system.md")),
        HumanMessage(content=render_prompt(PROMPTS_DIR / "observed_tool.user.md", variables)),
    ]
    try:
        refined = cast("ToolRefinement", await model.ainvoke(messages))
    except Exception as exc:
        rate_limit = rate_limit_error_from_exception(exc, source=f"smart_benign:observed_tool[{spec.name}]")
        if rate_limit is not None:
            raise rate_limit from exc
        raise
    extra = [e for e in dict.fromkeys(e.strip() for e in refined.extra_example_inputs) if e]
    extra = [e for e in extra if e not in spec.example_inputs][:_MAX_EXTRA_EXAMPLES]
    return spec.model_copy(
        update={"description": refined.description, "example_inputs": [*spec.example_inputs, *extra]}
    )


async def generate_for_tools(state: SynthState, tools: list[ToolSpec]) -> tuple[list[GeneratedRequest], list[str]]:
    """Run ``request_generator`` -> ``critic`` for ``tools`` only; return the kept rows and any errors.

    The profile's personas, role and out-of-scope list still apply, so the new rows match the rest of the
    suite. The critic dedupes within the new rows only.
    """
    if state.profile is None or not tools:
        return [], []
    scoped = state.model_copy(
        update={"profile": state.profile.model_copy(update={"tools": tools}), "requests": [], "errors": []}
    )
    generated = await _run(
        request_generator.build_request_generator_agent(),
        request_generator.RequestGeneratorState,
        request_generator.entry_adapter,
        request_generator.exit_adapter,
        scoped,
    )
    scoped = scoped.model_copy(update={"requests": generated.get("requests", [])})
    criticized = await _run(
        critic.build_critic_agent(), critic.CriticState, critic.entry_adapter, critic.exit_adapter, scoped
    )
    errors = [*generated.get("errors", []), *criticized.get("errors", [])]
    return list(criticized.get("requests", [])), errors


async def _run(agent: Any, state_type: Any, entry: Callable, exit_: Callable, state: SynthState) -> dict[str, Any]:
    """Invoke one compiled subgraph the way the parent graph's nodes do (``graph._wrap``)."""
    return cast("dict[str, Any]", exit_(state_type.model_validate(await agent.ainvoke(entry(state)))))


def _render_calls(calls: list[ObservedCall]) -> str:
    blocks = []
    for i, call in enumerate(calls, start=1):
        args = _clip(json.dumps(call.args, default=str), _ARGS_CHARS)
        blocks.append(f"{i}. request: {call.prompt}\n   args: {args}\n   result: {call.result_excerpt}")
    return "\n".join(blocks)


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"
