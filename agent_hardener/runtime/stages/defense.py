# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The defense stage: route attacks to defenders, run them, and aggregate their policy patches.

This is the *runner* for the defender phase — it composes the LLM router
(:class:`~agent_hardener.agents.defenders.defenders_manager.DefendersManager`, which stays in
``agents/defenders/``) with the per-defender invocation and the phase's events + artifact writes.

Why defenders diverge from the unified agent contract
------------------------------------------------------
Attackers, victims, and validators all conform to ``run(request, agent) -> AgentRunOutput``
(see :mod:`agent_hardener.models.contracts`). Defenders deliberately do not: they take a
:class:`~agent_hardener.models.DefenderInput` and return a :class:`~agent_hardener.models.DefenderOutput`,
and they are fanned in by :class:`DefendersManager` rather than invoked independently. This is
intentional, not an oversight — unlike the other roles, defenders *mutate shared state* (the one
victim policy / workflow), so an LLM router first decides which defenders each attack should reach
and the manager then aggregates their patches into a single coherent change. Keep this boundary:
``agents/defenders/`` holds defender implementations + the manager; this stage owns the wiring.
"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import TYPE_CHECKING

from agent_hardener.agents.defenders.defenders_manager import DefendersManager
from agent_hardener.agents.validators.smart_benign.validator import load_requests
from agent_hardener.events import EventType
from agent_hardener.loggers import using_current_agent
from agent_hardener.models import (
    CURRENT_POLICY_KEY,
    BenignRequest,
    DefenderOutput,
    DefendersManagerInput,
    DefendersManagerOutput,
    RelayVictimSpec,
)
from agent_hardener.runtime.agent_loader import load_run_callable
from agent_hardener.runtime.stages.context import DefenseResult
from agent_hardener.storage import to_jsonable
from agent_hardener.swarm_tracker import benign_profiles_dir, victim_active_state_dir

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import AgentConfig, DefenderInput, SessionConfig
    from agent_hardener.runtime.run_recorder import RunRecorder
    from agent_hardener.runtime.stages.agent_invoker import AgentInvoker
    from agent_hardener.runtime.stages.context import IterationContext

logger = logging.getLogger(__name__)


class DefenseStage:
    """Route each attack to matching defenders, run them, and return analyses + aggregated patches."""

    def __init__(
        self,
        config: SessionConfig,
        invoker: AgentInvoker,
        defenders_manager: DefendersManager | None = None,
    ) -> None:
        self.config = config
        self.invoker = invoker
        self.defenders_manager = defenders_manager or DefendersManager()

    async def run(self, ctx: IterationContext) -> DefenseResult:
        # No configured defenders → nothing to generate. Frozen "validate-only" runs use this (defenders: [])
        # to replay attacks + benign against a fixed victim without producing new mitigations. Returning early
        # avoids a wasted router LLM call (which would then fail to route and log a spurious critical error).
        if not self.config.defenders:
            return DefenseResult(analyses=[], policy_patches=[])
        recorder = ctx.recorder
        attacker_summaries = [attack.summary for attack in ctx.attacks if attack.summary]
        # Defenders operate on the run's victim-active-state copy of the workflow (defenders patch it
        # there; the victim-control adapter then deploys it), so they need the active-state workflow path.
        manager_context = self.invoker.request_context()
        if ctx.run_dir is not None and self.config.target.agent_relay_plugins is not None:
            active_state_config = victim_active_state_dir(ctx.run_dir) / self.config.target.agent_relay_plugins.name
            manager_context = {**manager_context, "agent_relay_plugins": str(active_state_config)}
        # The policy defender edits the current policy: feed it the run's active-state policy (seeded from the
        # inferred initial policy), preferring the per-run copy and falling back to the configured seed.
        policy_seed = self._policy_seed(ctx.run_dir)
        if policy_seed is not None:
            manager_context = {**manager_context, CURRENT_POLICY_KEY: policy_seed.read_text(encoding="utf-8")}
        if ctx.run_dir is not None and self.config.storage.victim_policy_path is not None:
            active_state_policy = victim_active_state_dir(ctx.run_dir) / self.config.storage.victim_policy_path.name
            manager_context = {**manager_context, "victim_policy_path": str(active_state_policy)}
        recorder.emit(
            EventType.ATTACKER_SUMMARIES_PREPARED,
            iteration=ctx.iteration,
            attack_count=len(ctx.attacks),
            summary_count=len(attacker_summaries),
        )
        # The defenders manager routes each attack to the defenders whose capabilities match it
        # (by LLM), runs them with per-defender validation feedback, and returns their analyses.
        manager_input = DefendersManagerInput(
            round_id=recorder.round_id,
            target=self.config.target,
            context=manager_context,
            attacks=ctx.attacks,
            attacker_summaries=attacker_summaries,
            benign_requests=self._load_benign_requests(),
            available_defenders=self.config.defenders,
            relay_victim_spec=self._relay_victim_spec(),
        )
        phase_started_at = perf_counter()
        recorder.emit(
            EventType.PHASE_STARTED, iteration=ctx.iteration, phase="defenders", count=len(self.config.defenders)
        )
        try:
            manager_output = await self.defenders_manager.run(
                input_data=manager_input,
                run_defender_cb=lambda agent, inp: self._run_defender(agent, inp, recorder, ctx.iteration),
                validation_feedback=ctx.validation_feedback or None,
            )
        except Exception as exc:
            recorder.logger.exception("defenders manager failed critically: %s", exc)
            manager_output = DefendersManagerOutput(ok=False, error=str(exc))
        defenders = manager_output.analyses
        phase_duration = perf_counter() - phase_started_at
        recorder.emit(
            EventType.PHASE_COMPLETED,
            iteration=ctx.iteration,
            phase="defenders",
            count=len(defenders),
            routed_count=len(manager_output.routed_agent_ids),
            ok=manager_output.ok,
            duration_seconds=round(phase_duration, 6),
        )
        policy_patches = [
            policy_patch for defender in defenders if defender.ok for policy_patch in defender.policy_patches
        ]
        recorder.emit(
            EventType.POLICY_PATCHES_AGGREGATED,
            iteration=ctx.iteration,
            defender_count=len(defenders),
            patch_count=len(policy_patches),
        )
        # Full serialized defenders + aggregated patches so a live observer can highlight per-defender
        # counts AND render the defender's reasoning at stage end (verbose). Reuses final_log renderers.
        recorder.emit(
            EventType.DEFENDER_SUMMARY,
            iteration=ctx.iteration,
            defenders=[to_jsonable(defender) for defender in defenders],
            policy_patches=to_jsonable(policy_patches),
        )
        # Per-defender transcript row (attack prompt -> the defender's reasoning) for the UI's agent view.
        for defender in defenders:
            recorder.emit(
                EventType.AGENT_EXCHANGE,
                agent_id=defender.agent_id,
                agent_name=defender.agent_name,
                agent_role="defender",
                phase="defenders",
                iteration=ctx.iteration,
                request=defender.attack_prompt or "",
                response=defender.summary,
                label="analysis",
                ok=bool(defender.ok),
            )
        if ctx.iteration_dir is not None:
            ctx.store.write_component(ctx.iteration_dir, "defenders_manager", "manager", manager_output)
            for analysis in defenders:
                ctx.store.write_component(ctx.iteration_dir, "defender", analysis.agent_name, analysis)
        return DefenseResult(analyses=defenders, policy_patches=policy_patches)

    def _policy_seed(self, run_dir: Path | None) -> Path | None:
        """Return the current policy file the defender's ``current_policy`` is seeded from.

        Prefers the run's active-state copy (run-scoped, evolves with the run); falls back to the configured
        initial policy (the inferred policy the runner seeds). Returns None when there is no policy file.
        """
        policy_path = self.config.storage.victim_policy_path
        if policy_path is None:
            return None
        if run_dir is not None:
            active = victim_active_state_dir(run_dir) / policy_path.name
            if active.exists():
                return active
        return policy_path if policy_path.exists() else None

    def _relay_victim_spec(self) -> RelayVictimSpec | None:
        """The run's ``victim_control.config["relay_victim"]`` block, typed, if this target has one."""
        raw = self.config.victim_control.config.get("relay_victim")
        return RelayVictimSpec.model_validate(raw) if raw is not None else None

    def _load_benign_requests(self) -> list[BenignRequest]:
        """Load the smart-benign suite synthesized pre-flight for this target.

        Reuses the suite that ``agent-hardener synth-benign`` / the smart benign validator persists at
        ``<storage_root>/benign_profiles/<target>/requests.csv`` so the defenders manager protects the
        same baseline requests the benign validator replays. Returns an empty list when no suite has
        been synthesized yet (the manager then runs without a benign baseline).
        """
        csv_path = benign_profiles_dir(self.config.storage.root_dir, self.config.target.name) / "requests.csv"
        if not csv_path.exists():
            return []
        return [BenignRequest(payload=request.payload) for request in load_requests(csv_path)]

    async def _run_defender(
        self,
        agent: AgentConfig,
        defender_input: DefenderInput,
        recorder: RunRecorder,
        iteration_number: int,
    ) -> DefenderOutput:
        """Invoke one defender via its DefenderInput contract (the manager's run callback)."""
        payload = {
            "iteration": iteration_number,
            "phase": "defenders",
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent.role,
            "validator_kind": None,
        }
        recorder.emit(EventType.AGENT_STARTED, **payload)
        started_at = perf_counter()
        try:
            run_callable = load_run_callable(agent.implementation)
            with using_current_agent(payload):
                if asyncio.iscoroutinefunction(run_callable):
                    result = await asyncio.wait_for(run_callable(defender_input), timeout=agent.timeout_seconds)
                else:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(run_callable, defender_input), timeout=agent.timeout_seconds
                    )
            output = (
                result
                if isinstance(result, DefenderOutput)
                else DefenderOutput(ok=False, error_message=f"unexpected return type: {type(result).__name__}")
            )
        except Exception as exc:
            duration_seconds = perf_counter() - started_at
            recorder.logger.exception(
                "defender stop name=%s id=%s ok=False duration_seconds=%.3f error=%s",
                agent.name,
                agent.agent_id,
                duration_seconds,
                str(exc),
            )
            recorder.emit(
                EventType.AGENT_FAILED, **payload, duration_seconds=round(duration_seconds, 6), error=str(exc)
            )
            return DefenderOutput(ok=False, error_message=str(exc))

        duration_seconds = perf_counter() - started_at
        recorder.emit(
            EventType.AGENT_COMPLETED,
            **payload,
            ok=output.ok,
            duration_seconds=round(duration_seconds, 6),
            summary="",
            error=output.error_message,
        )
        return output
