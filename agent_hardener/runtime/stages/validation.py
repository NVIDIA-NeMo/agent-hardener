# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The validation stage: typed per-agent fan-out over the attack and benign validators."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, cast

from agent_hardener.concurrency import concurrency_limit, gather_limited_ordered
from agent_hardener.events import EventType
from agent_hardener.models import AgentRunInput, ValidatorReport
from agent_hardener.runtime.agent_loader import failure_output
from agent_hardener.runtime.stages.agent_invoker import all_ok

if TYPE_CHECKING:
    from agent_hardener.models import AgentConfig, SessionConfig, ValidatorKind, VictimResult
    from agent_hardener.runtime.stages.agent_invoker import AgentInvoker
    from agent_hardener.runtime.stages.context import DefenseResult, IterationContext


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
            if isinstance(result, BaseException):
                validators.append(cast("ValidatorReport", failure_output(agent, request, result)))
            elif isinstance(result, ValidatorReport):
                validators.append(result)
            else:
                recorder.logger.warning("expected ValidatorReport from validator agent, got %s", type(result).__name__)
                err = TypeError(f"validator agent returned {type(result).__name__}, expected ValidatorReport")
                validators.append(cast("ValidatorReport", failure_output(agent, request, err)))
        phase_duration = perf_counter() - phase_started_at
        recorder.emit(
            EventType.PHASE_COMPLETED,
            iteration=ctx.iteration,
            phase="validators",
            count=len(validators),
            ok=all_ok(validators),
            duration_seconds=round(phase_duration, 6),
        )
        if ctx.iteration_dir is not None:
            for report in validators:
                ctx.store.write_component(ctx.iteration_dir, "validator", report.agent_name, report)
        return validators
