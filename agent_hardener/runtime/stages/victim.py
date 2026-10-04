# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The victim stage: run the (post-defense) victim agent once for the iteration."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, cast

from agent_hardener.events import EventType
from agent_hardener.models import AgentRunInput, VictimResult
from agent_hardener.runtime.agent_loader import failure_output
from agent_hardener.runtime.stages.agent_invoker import short_summary

if TYPE_CHECKING:
    from agent_hardener.models import SessionConfig
    from agent_hardener.runtime.stages.agent_invoker import AgentInvoker
    from agent_hardener.runtime.stages.context import DefenseResult, IterationContext


class VictimStage:
    """Invoke the victim agent for one iteration."""

    def __init__(self, config: SessionConfig, invoker: AgentInvoker) -> None:
        self.config = config
        self.invoker = invoker

    async def run(self, ctx: IterationContext, defense: DefenseResult) -> VictimResult:
        recorder = ctx.recorder
        victim_request = AgentRunInput(
            round_id=recorder.round_id,
            target=self.config.target,
            context=self.invoker.request_context(),
            iteration=ctx.iteration,
            attacks=ctx.attacks,
            defender_analyses=defense.analyses,
            policy_patches=defense.policy_patches,
        )
        phase_started_at = perf_counter()
        recorder.emit(EventType.PHASE_STARTED, iteration=ctx.iteration, phase="victim", count=1)
        victim_output = await self.invoker.run_agent(
            self.config.victim, victim_request, "victim", recorder.mission_id, recorder.logger, recorder.events
        )
        if isinstance(victim_output, VictimResult):
            victim = victim_output
        else:
            recorder.logger.warning("expected VictimResult from victim agent, got %s", type(victim_output).__name__)
            err = TypeError(f"victim agent returned {type(victim_output).__name__}, expected VictimResult")
            victim = cast("VictimResult", failure_output(self.config.victim, victim_request, err))
        if not victim.ok:
            recorder.logger.warning(
                "victim returned failure status iteration=%s agent=%s error=%s summary=%s",
                ctx.iteration,
                self.config.victim.name,
                victim.error,
                short_summary(victim.summary),
            )
            recorder.emit(
                EventType.VICTIM_WARNING,
                iteration=ctx.iteration,
                agent_id=self.config.victim.agent_id,
                agent_name=self.config.victim.name,
                ok=victim.ok,
                summary=victim.summary,
                error=victim.error,
            )
        phase_duration = perf_counter() - phase_started_at
        recorder.emit(
            EventType.PHASE_COMPLETED,
            iteration=ctx.iteration,
            phase="victim",
            count=1,
            ok=victim.ok,
            duration_seconds=round(phase_duration, 6),
        )
        return victim
