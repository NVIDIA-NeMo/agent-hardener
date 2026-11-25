# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The validation stage: typed per-agent fan-out over the attack and benign validators."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, cast

from agent_hardener.agents.validators.trace_step import open_window
from agent_hardener.concurrency import concurrency_limit, gather_limited_ordered
from agent_hardener.errors import GuardrailJudgeError, VictimUnavailableError
from agent_hardener.events import EventType
from agent_hardener.models import AgentRunInput, ValidatorReport
from agent_hardener.relay_plugin.config import guarded_tools, read_plugins_document
from agent_hardener.runtime.agent_loader import failure_output
from agent_hardener.runtime.stages.agent_invoker import all_ok
from agent_hardener.traces.inspect import same_tool

if TYPE_CHECKING:
    import logging

    from agent_hardener.models import AgentConfig, SessionConfig, ValidatorKind, VictimResult
    from agent_hardener.runtime.stages.agent_invoker import AgentInvoker
    from agent_hardener.runtime.stages.context import DefenseResult, IterationContext
    from agent_hardener.traces import AgentTrace

#: Above this share of judged calls failing, the judge is down rather than flaky: a timeout or a rate
#: limit hits a few calls, a bad key or an unreachable endpoint hits them all.
JUDGE_ERROR_RATE_LIMIT = 0.5
#: Fewer failures than this never fail a round, so one blip in a small round is not read as a dead judge.
JUDGE_ERROR_MIN_COUNT = 3


class ValidationStage:
    """Run every attack and benign validator against the victim's post-defense behavior."""

    def __init__(self, config: SessionConfig, invoker: AgentInvoker) -> None:
        self.config = config
        self.invoker = invoker

    async def run(self, ctx: IterationContext, defense: DefenseResult, victim: VictimResult) -> list[ValidatorReport]:
        recorder = ctx.recorder
        validator_specs: list[tuple[AgentConfig, ValidatorKind]] = [
            *[(agent, "attack") for agent in self.config.attack_validators],
            *[(agent, "benign") for agent in self.config.benign_validators],
        ]
        # Opened before any validator runs, so it holds exactly the validators' traffic.
        window = await open_window(self.invoker.trace_provider)
        phase_started_at = perf_counter()
        recorder.emit(
            EventType.PHASE_STARTED,
            iteration=ctx.iteration,
            phase="validators",
            count=len(validator_specs),
            attack_validator_count=len(self.config.attack_validators),
            benign_validator_count=len(self.config.benign_validators),
        )
        validator_agents = [agent for agent, _kind in validator_specs]
        validator_requests = [
            AgentRunInput(
                round_id=recorder.round_id,
                target=self.config.target,
                context=self.invoker.request_context(),
                iteration=ctx.iteration,
                attacks=ctx.attacks,
                defender_analyses=defense.analyses,
                policy_patches=defense.policy_patches,
                victim_result=victim,
                validator_kind=kind,
            )
            for _agent, kind in validator_specs
        ]
        results = await gather_limited_ordered(
            list(zip(validator_agents, validator_requests, strict=True)),
            concurrency_limit(
                self.config.run.concurrency.validators,
                "AGENT_HARDENER_VALIDATOR_CONCURRENCY",
            ),
            lambda item: self.invoker.run_agent(
                item[0], item[1], "validators", recorder.mission_id, recorder.logger, recorder.events
            ),
            return_exceptions=True,
        )
        validators: list[ValidatorReport] = []
        for agent, request, result in zip(validator_agents, validator_requests, results, strict=True):
            if isinstance(result, VictimUnavailableError):
                raise result  # a dead victim must abort the run, not be scored as a failed validator
            if isinstance(result, BaseException):
                validators.append(cast("ValidatorReport", failure_output(agent, request, result)))
            elif isinstance(result, ValidatorReport):
                validators.append(result)
            else:
                recorder.logger.warning("expected ValidatorReport from validator agent, got %s", type(result).__name__)
                err = TypeError(f"validator agent returned {type(result).__name__}, expected ValidatorReport")
                validators.append(cast("ValidatorReport", failure_output(agent, request, err)))
        guarded = guarded_tools(read_plugins_document(self.config.target.agent_relay_plugins))
        unscored, judged = judge_failures(await window.traces(), guarded) if window is not None else (0, 0)
        phase_duration = perf_counter() - phase_started_at
        recorder.emit(
            EventType.PHASE_COMPLETED,
            iteration=ctx.iteration,
            phase="validators",
            count=len(validators),
            ok=all_ok(validators),
            duration_seconds=round(phase_duration, 6),
            judge_errors=unscored,
            judged_calls=judged,
        )
        if ctx.iteration_dir is not None:
            for report in validators:
                ctx.store.write_component(ctx.iteration_dir, "validator", report.agent_name, report)
        _check_judge(unscored, judged, recorder.logger)
        return validators


def judge_failures(traces: list[AgentTrace], guarded: set[str]) -> tuple[int, int]:
    """How many calls the guardrails' judge could not score, out of the calls it was asked to score.

    Every call to a guarded tool is judged, so the run's guardrail set gives the exact denominator.
    """
    unscored = sum(len(trace.judge_error_tools) for trace in traces)
    judged = sum(
        1 for trace in traces for call in trace.tool_calls if any(same_tool(call.name, tool) for tool in guarded)
    )
    return unscored, max(judged, unscored)


def _check_judge(unscored: int, judged: int, logger: logging.Logger) -> None:
    """Fail the round when the judge is down; otherwise say how many calls it let through unscored."""
    if not unscored:
        return
    message = f"guardrail safety judge failed on {unscored}/{judged} guarded tool calls, which were allowed unscored"
    if unscored >= JUDGE_ERROR_MIN_COUNT and unscored / judged > JUDGE_ERROR_RATE_LIMIT:
        raise GuardrailJudgeError(message)
    logger.warning(message)
