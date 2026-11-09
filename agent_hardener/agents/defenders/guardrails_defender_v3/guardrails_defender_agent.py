# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point of the v3 guardrails defender: one attack in, one targeted rail out.

Every rail is grounded in the attack's execution trace, so an attack that arrives without one is an
error, never a silent skip: the victim must run with NeMo Relay, and a replay must load the traces
recorded with its hitlog.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import TYPE_CHECKING, Any

from agent_hardener.models import DefenderInput, DefenderOutput
from agent_hardener.traces.attacks import attack_key

from .graph import build_graph
from .llm import GRADER_TEMPERATURE, get_llm
from .observability import OUTCOME_FILENAME, SUMMARY_FILENAME, Recorder, artifacts_dir, render_summary
from .schemas import Outcome
from .state import DefenderState

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

logger = logging.getLogger("agent_hardener.session")

#: Why an attack without an execution trace gets no rail.
NO_TRACE_REASON = (
    "no execution trace for this attack: the victim must be instrumented with NeMo Relay, and a replay "
    "must load the attack traces recorded with its hitlog"
)


def run(
    defender_input: DefenderInput, llm: BaseChatModel | None = None, grader: BaseChatModel | None = None
) -> DefenderOutput:
    """Find where the attack did its harm in its trace, and apply a rail there."""
    attack = attack_name(defender_input)
    recorder = Recorder(artifacts_dir(defender_input, attack))
    recorder.write("input", _input_record(defender_input))
    if defender_input.attack_trace is None:
        outcome = Outcome(attack=attack, status="error", reason=NO_TRACE_REASON)
        recorder.write_file(OUTCOME_FILENAME, outcome.model_dump_json(indent=2))
        recorder.write_file(SUMMARY_FILENAME, render_summary(outcome, None, [], []))
        logger.error("defender v3 %s: %s", attack, NO_TRACE_REASON)
        return DefenderOutput(ok=False, error_message=f"defender v3 failed: {NO_TRACE_REASON}")
    if llm is None:
        llm = get_llm()
    if grader is None:
        grader = get_llm(temperature=GRADER_TEMPERATURE)
    graph = build_graph(llm, grader, recorder)
    state = DefenderState(input=defender_input, attack=attack)
    final: dict[str, Any] = {}
    try:
        final = graph.invoke(state)
        outcome: Outcome = final["outcome"]
        output: DefenderOutput = final["output"]
    except Exception as exc:
        logger.exception("defender v3 failed on %s", attack)
        outcome = Outcome(attack=attack, status="error", reason=f"{type(exc).__name__}: {exc}")
        output = DefenderOutput(ok=False, error_message=f"defender v3 failed: {outcome.reason}")
    recorder.write_file(OUTCOME_FILENAME, outcome.model_dump_json(indent=2))
    recorder.write_file(
        SUMMARY_FILENAME,
        render_summary(
            outcome, final.get("vulnerability_analysis"), final.get("drafts") or [], final.get("grades") or []
        ),
    )
    logger.info("defender v3 %s: %s %s", attack, outcome.status, outcome.reason or outcome.guardrail_name or "")
    return output


def attack_name(defender_input: DefenderInput) -> str:
    """A short, stable, filesystem-safe name: the attacked tool plus a hash of the normalized prompt."""
    digest = hashlib.sha1(attack_key(defender_input.attack_prompt).encode("utf-8"), usedforsecurity=False)
    tool = re.sub(r"[^A-Za-z0-9_-]+", "-", defender_input.attacked_tool.removeprefix("functions.")) or "attack"
    return f"{tool}-{digest.hexdigest()[:10]}"


def _input_record(defender_input: DefenderInput) -> dict[str, Any]:
    return {
        "attack_prompt": defender_input.attack_prompt,
        "agent_response": defender_input.agent_response,
        "attacked_tool": defender_input.attacked_tool,
        "round_id": defender_input.context.get("round_id"),
        "benign_suite": defender_input.benign_suite or defender_input.benign_requests,
        "attack_trace": defender_input.attack_trace,
    }
