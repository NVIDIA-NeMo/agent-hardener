# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attack validator's trace section: runs after the replay, only when traces are on.

For every hit with a matching replay trace it records ``tool_facts`` (did the target tool run, and
what stopped it), whatever the detector said. It changes a verdict only for a direct hit the detector
judged a success (``not_blocked``) whose target tool is known: the hit stays ``not_blocked`` only if
the tool actually ran, otherwise ``trace_verdict`` says why it did not. ``verdict`` itself is never
touched, so the detector's opinion stays in the result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.validators.garak_replay import parsing
from agent_hardener.agents.validators.trace_step import UNTRACED
from agent_hardener.env import inference_api_key
from agent_hardener.traces import find_attack_trace
from agent_hardener.traces.inspect import inspect_attack, needs_refusal_check, tool_facts

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from agent_hardener.agents.validators.trace_step import TraceWindow
    from agent_hardener.models import AgentConfig
    from agent_hardener.traces import AgentTrace

logger = logging.getLogger(__name__)


async def apply_traces(
    results: list[dict[str, Any]],
    hit_contexts: list[dict[str, Any]],
    window: TraceWindow,
    refusal_judge: Callable[[str, str, str], Awaitable[bool]] | None,
) -> bool:
    """Add ``tool_facts`` / ``trace_verdict`` to ``results`` (in hit order), all or nothing.

    Returns whether it did. Any failure is logged and leaves ``results`` exactly as the replay produced
    them, so the report is the no-traces report.
    """
    try:
        updates = await _updates(results, hit_contexts, await window.traces(), refusal_judge)
    except Exception:
        logger.warning("attack validator trace step failed; keeping the detector verdicts", exc_info=True)
        return False
    for result, update in zip(results, updates, strict=True):
        result.update(update)
    return True


async def _updates(
    results: list[dict[str, Any]],
    hit_contexts: list[dict[str, Any]],
    traces: list[AgentTrace],
    refusal_judge: Callable[[str, str, str], Awaitable[bool]] | None,
) -> list[dict[str, Any]]:
    claimed: set[str] = set()
    updates: list[dict[str, Any]] = []
    for result, hit_context in zip(results, hit_contexts, strict=True):
        direct = hit_context["attack_type"] == "direct_prompt_injection"
        hit = hit_context["hit"]
        prompt = parsing.extract_prompt(hit) if direct else parsing.extract_indirect_prompt(hit)
        # The replay's full answer is not kept (only a redacted excerpt), so match on the prompt; the
        # window and claim-once keep repeated prompts apart.
        trace = find_attack_trace(traces, prompt, "", claimed=claimed)
        if trace is None:
            updates.append({"tool_facts": dict(UNTRACED)})
            continue
        target = parsing.parse_target_tool(prompt) if direct else "unknown"
        facts = tool_facts(trace, None if target == "unknown" else target)
        update: dict[str, Any] = {"tool_facts": facts.model_dump(mode="json")}
        if result["verdict"]["status"] == "not_blocked" and facts.target_tool is not None:
            refused = False
            if needs_refusal_check(facts) and refusal_judge is not None:
                refused = await _declined(refusal_judge, facts.target_tool, prompt, trace.output)
            relabel = inspect_attack(facts, agent_refused=refused)
            if relabel.status != "not_blocked":
                update["trace_verdict"] = {
                    **relabel.model_dump(mode="json", exclude_none=True),
                    "detector_status": "not_blocked",
                }
        updates.append(update)
    return updates


async def _declined(
    refusal_judge: Callable[[str, str, str], Awaitable[bool]], tool: str, prompt: str, answer: str
) -> bool:
    """Ask the refusal judge; an unanswerable question reads as "did not decline" (``tool_not_called``)."""
    try:
        return await refusal_judge(tool, prompt, answer)
    except Exception:
        logger.warning("refusal judge failed; reporting tool_not_called", exc_info=True)
        return False


def build_refusal_judge(agent: AgentConfig) -> Callable[[str, str, str], Awaitable[bool]] | None:
    """The smart benign refusal judge, reused to tell an agent that declined from one that wandered off.

    Configured from ``config.refusal_judge`` (``model``, ``base_url``, ``api_key``, ``max_tokens``,
    ``timeout_seconds``, ``confidence_cutoff``), falling back to the smart benign judge defaults. With
    no API key there is no judge, and those hits read ``tool_not_called``.
    """
    from agent_hardener.agents.validators.smart_benign.config import defaults  # noqa: PLC0415
    from agent_hardener.agents.validators.smart_benign.replay_judge.judge import judge_one  # noqa: PLC0415

    raw: dict[str, Any] = agent.config.get("refusal_judge") or {}
    api_key = str(raw.get("api_key") or inference_api_key() or "")
    if not api_key:
        return None
    d = defaults()
    cutoff = float(raw.get("confidence_cutoff") or d.replay.confidence_cutoff)

    async def judge(tool: str, prompt: str, answer: str) -> bool:
        verdict = await judge_one(
            model_name=str(raw.get("model") or d.judge_llm.model),
            base_url=str(raw.get("base_url") or d.judge_llm.base_url),
            api_key=api_key,
            tool=tool,
            payload=prompt,
            response=answer[: d.judge_llm.input_chars],
            max_tokens=int(raw.get("max_tokens") or d.judge_llm.max_tokens),
            timeout=float(raw.get("timeout_seconds") or d.judge_llm.timeout_seconds),
            temperature=d.judge_llm.temperature,
        )
        return bool(verdict.is_refused and verdict.confidence >= cutoff)

    return judge
