# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent Hardener session orchestration."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from agent_hardener.agents.attackers.preloaded import load_preloaded_attack_traces
from agent_hardener.agents.victims.openshell_victim import contract_for
from agent_hardener.events import EventType
from agent_hardener.ids import format_round_id, generate_round_id
from agent_hardener.models import (
    AttackRecord,
    RoundIterationReport,
    RoundReport,
    SessionConfig,
    ValidationFeedback,
)
from agent_hardener.runtime.adapters import (
    AgentRunner,
    RoutingAgentRunner,
    Uploader,
    build_uploaders,
)
from agent_hardener.runtime.feedback import build_validation_feedback
from agent_hardener.runtime.round_store import RoundStore
from agent_hardener.runtime.run_recorder import RunRecorder
from agent_hardener.runtime.stages import (
    AgentInvoker,
    AttackStage,
    DefenseStage,
    DeployStage,
    IterationContext,
    ValidationStage,
    VictimStage,
)
from agent_hardener.swarm_tracker import (
    create_round_dir,
    create_run_dir,
    update_victim_active_state,
    write_init_files,
    write_run_config,
)
from agent_hardener.traces import trace_clock, traces_since
from agent_hardener.traces.attacks import ATTACK_TRACES_FILENAME, by_prompt, collect_attack_traces, dump_recording
from agent_hardener.traces.replay_attacks import replay_attacks_for_traces

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from agent_hardener.agents.defenders.defenders_manager import DefendersManager
    from agent_hardener.traces import AgentTrace, TraceProvider
    from agent_hardener.traces.attacks import RecordedAttackTrace


class Orchestrator:
    """Singleton-friendly orchestrator configured from a session config."""

    def __init__(
        self,
        config: SessionConfig,
        agent_adapter: AgentRunner | None = None,
        openshell_uploader: Uploader | None = None,
        relay_uploader: Uploader | None = None,
        round_id_factory: Callable[[], str] = generate_round_id,
        mission_id_factory: Callable[[], str] = generate_round_id,
        defenders_manager: DefendersManager | None = None,
        verify_tool_path: Callable[[], None] | None = None,
        sync_atof: Callable[[], None] | None = None,
        trace_provider: TraceProvider | None = None,
        replay_attacks: Callable[..., Awaitable[list[RecordedAttackTrace]]] = replay_attacks_for_traces,
    ) -> None:
        self.config = config
        # The victim's traces: recorded for every round's attacks, and handed to every agent invoked
        # (an agent that does not read them ignores them). ``None`` only for a plain HTTP target.
        self.trace_provider = trace_provider
        # Called once, after the first round's attacks. Deliberately a hook rather than logic here:
        # proving tool calls reach Relay needs the victim's telemetry, which only the runner can pull
        # out of the sandbox. See agent_hardener.preflight.relay.check_tool_path.
        self.verify_tool_path = verify_tool_path
        self.sync_atof = sync_atof
        self._tool_path_verified = False
        # A replay sends the attacks with no recorded trace to the agent once, in its first round, and
        # reuses what that produced: later rounds' agent already carries rails written from those traces.
        self.replay_attacks = replay_attacks
        self._replay_traces: dict[str, AgentTrace] | None = None
        self.round_id_factory = round_id_factory
        self.mission_id_factory = mission_id_factory

        if openshell_uploader is not None and relay_uploader is not None:
            openshell, nat = openshell_uploader, relay_uploader
        else:
            built_openshell, built_relay = build_uploaders(config.victim_control)
            openshell = openshell_uploader or built_openshell
            nat = relay_uploader or built_relay
        invoker = AgentInvoker(config, agent_adapter or RoutingAgentRunner(), trace_provider)

        self.attack_stage = AttackStage(config, invoker)
        self.defense = DefenseStage(config, invoker, defenders_manager)
        self.deploy = DeployStage(config, openshell, nat)
        self.victim_stage = VictimStage(config, invoker)
        self.validation_stage = ValidationStage(config, invoker)

    async def run_rounds(
        self, rounds: int | None = None, mission_id: str | None = None, run_id: str | None = None
    ) -> list[RoundReport]:
        """Run hardening rounds until the count is reached."""
        effective_max = rounds if rounds is not None else self.config.run.rounds
        if effective_max < 1:
            return []

        effective_mission_id = mission_id or self.mission_id_factory()
        run_dir, starting_round = self._prepare_run(run_id)

        reports: list[RoundReport] = []
        while len(reports) < effective_max:
            round_number = starting_round + len(reports)
            reports.append(
                await self.run_round(
                    mission_id=effective_mission_id,
                    round_dir=create_round_dir(run_dir, round_number),
                    run_dir=run_dir,
                    round_id=format_round_id(round_number),
                )
            )
            if len(reports) < effective_max and self.config.run.round_interval_seconds:
                await asyncio.sleep(self.config.run.round_interval_seconds)
        return reports

    def _prepare_run(self, run_id: str | None) -> tuple[Path, int]:
        """Create the run-log tree, persist the config, and seed victim state for round 1."""
        storage = self.config.storage
        run_id = run_id or storage.run_id or self.mission_id_factory()
        run_dir = create_run_dir(storage.root_dir, run_id)
        write_run_config(run_dir, self.config)
        starting_round = storage.round_number or 1
        if starting_round == 1:
            write_init_files(run_dir, storage.victim_policy_path, storage.victim_relay_plugins_path)
            update_victim_active_state(run_dir, storage.victim_policy_path, storage.victim_relay_plugins_path)
        return run_dir, starting_round

    async def run_round(
        self,
        mission_id: str | None = None,
        round_dir: Path | None = None,
        run_dir: Path | None = None,
        round_id: str | None = None,
    ) -> RoundReport:
        """Run one full security war-game orchestration round."""
        round_id = round_id or self.round_id_factory()
        started_at = datetime.now(UTC)
        round_dir = round_dir or self.config.storage.root_dir
        store = RoundStore(round_dir)
        with RunRecorder(round_id, mission_id, round_dir) as recorder:
            recorder.emit(EventType.ROUND_STARTED, storage_dir=str(round_dir))

            # Attackers run once for the round; each iteration then hardens against them.
            provider = self.trace_provider if self.config.attackers else None
            anchor = await _trace_anchor(provider, recorder) if provider is not None else None
            attacks = await self.attack_stage.run(recorder, store)
            # First real tool traffic of the run: the earliest point where "do tool calls reach
            # Relay?" is answerable. Before authoring guardrails, because a victim that fails this
            # cannot be guarded by anything they write.
            #
            # Only when an attacker actually ran. A replay feeds the defenders recorded hits without
            # invoking the victim at all, so there is no traffic to judge and a passing victim would
            # be failed for someone else's silence (a replay that re-sends attacks checks below).
            if self.verify_tool_path is not None and not self._tool_path_verified and self.config.attackers:
                self.verify_tool_path()
                self._tool_path_verified = True
            # Recorded whenever traces are on and live attackers ran. A replay loads its recording, and
            # replays the attacks the recording has no trace for to obtain theirs.
            recorded = _replayed_attack_traces(self.config, recorder)
            replay_provider = self._replay_provider()
            attack_traces = {
                **(
                    await self._replay_for_traces(replay_provider, recorder, store, attacks, recorded)
                    if replay_provider is not None
                    else {}
                ),
                **recorded,
                **(await _collect_attack_traces(provider, recorder, store, attacks, anchor) if provider else {}),
            }
            iterations, success = await self._run_iterations(recorder, store, attacks, run_dir, attack_traces)

            # Pull the victim's telemetry once the round's traffic is done, whatever produced it.
            # The tool-path check above is the only other fetch and it is gated on live attackers, so
            # without this a replay run's host copy stayed frozen at the startup probe: the round's
            # own tool calls — made by the validators against the victim — were never collected, and
            # every ATOF-derived answer came back empty rather than unknown.
            if self.sync_atof is not None:
                self.sync_atof()

            report = RoundReport(
                round_id=round_id,
                mission_id=mission_id,
                started_at=started_at,
                ended_at=datetime.now(UTC),
                success=success,
                attacks=attacks,
                iterations=iterations,
                storage_dir=round_dir,
            )
            report_path = store.write_round_json("report.json", report)
            recorder.emit(EventType.REPORT_WRITTEN, path=str(report_path), success=success)
            recorder.emit(EventType.ROUND_COMPLETED, success=success, iterations=len(iterations))
            return report

    def _replay_provider(self) -> TraceProvider | None:
        """The trace provider in a replay run (recorded hits, no live attacker), else ``None``."""
        replay = not self.config.attackers and bool(self.config.preloaded_attacks)
        return self.trace_provider if replay else None

    async def _replay_for_traces(
        self,
        provider: TraceProvider,
        recorder: RunRecorder,
        store: RoundStore,
        attacks: list[AttackRecord],
        recorded: dict[str, AgentTrace],
    ) -> dict[str, AgentTrace]:
        """Traces for the replayed attacks that have no recorded one, obtained by sending them to the agent.

        Done once per run, in its first round, and written to that round's ``attack_traces.json``; later
        rounds reuse the result. A failure never fails the round: those attacks reach the defenders without
        a trace, and the defender reports it.
        """
        if self._replay_traces is not None:
            return self._replay_traces
        try:
            replayed = await self.replay_attacks(
                attacks,
                recorded=recorded,
                provider=provider,
                contract=contract_for(self.config.target.base_url, self.config.victim),
            )
        except Exception:
            recorder.logger.warning("could not replay the attacks for their traces", exc_info=True)
            replayed = []
        self._replay_traces = by_prompt(replayed)
        if replayed:
            # The replay was the run's first tool traffic: prove it reaches Relay before writing guardrails.
            if self.verify_tool_path is not None and not self._tool_path_verified:
                self.verify_tool_path()
                self._tool_path_verified = True
            path = store.write_round_json(ATTACK_TRACES_FILENAME, dump_recording(replayed))
            recorder.emit(
                EventType.ARTIFACT_WRITTEN, artifact=ATTACK_TRACES_FILENAME, path=str(path), count=len(replayed)
            )
        return self._replay_traces

    async def _run_iterations(
        self,
        recorder: RunRecorder,
        store: RoundStore,
        attacks: list[AttackRecord],
        run_dir: Path | None,
        attack_traces: dict[str, AgentTrace] | None = None,
    ) -> tuple[list[RoundIterationReport], bool]:
        """Run defend → deploy → victim → validate iterations until success or the retry limit.

        One iteration unless ``run.retries_enabled`` (the feature flag) is on; see :class:`RunSettings`.
        """
        iterations: list[RoundIterationReport] = []
        validation_feedback: dict[str, ValidationFeedback] = {}
        retry_limit = self.config.run.effective_retry_limit
        if self.config.run.retry_limit and not self.config.run.retries_enabled:
            recorder.logger.warning(
                "run.retry_limit=%d is ignored: retries are disabled (set run.retries_enabled to turn them on)",
                self.config.run.retry_limit,
            )
        for iteration_number in range(1, retry_limit + 2):
            recorder.emit(EventType.ITERATION_STARTED, iteration=iteration_number, retry_limit=retry_limit)
            iteration_dir = store.iteration_dir(iteration_number)
            ctx = IterationContext(
                recorder=recorder,
                store=store,
                iteration=iteration_number,
                attacks=attacks,
                iteration_dir=iteration_dir,
                run_dir=run_dir,
                validation_feedback=validation_feedback or None,
                attack_traces=attack_traces or {},
            )
            report = await self._run_iteration(ctx)
            iterations.append(report)
            recorder.emit(EventType.ITERATION_COMPLETED, iteration=iteration_number, success=report.success)
            if report.success:
                return iterations, True
            validation_feedback = build_validation_feedback(report)
        return iterations, False

    async def _run_iteration(self, ctx: IterationContext) -> RoundIterationReport:
        """Run one attempt of the defend → deploy → victim → validate pipeline."""
        defense = await self.defense.run(ctx)
        victim_control = await self.deploy.run(ctx, defense)
        victim = await self.victim_stage.run(ctx, defense)
        validators = await self.validation_stage.run(ctx, defense, victim)
        return RoundIterationReport(
            iteration=ctx.iteration,
            defenders=defense.analyses,
            policy_patches=defense.policy_patches,
            victim_control=victim_control,
            victim=victim,
            validators=validators,
            success=victim_control.ok and all(validator.ok for validator in validators),
        )


def _replayed_attack_traces(config: SessionConfig, recorder: RunRecorder) -> dict[str, AgentTrace]:
    """Traces recorded by an earlier run, for a replay (``--replay``; ``--with-traces PATH`` to pick the file).

    The CLI already validated the file, so a failure here (it changed since) is logged and the replayed hits
    are treated as unrecorded: :meth:`Orchestrator._replay_for_traces` then sends them to the agent.
    """
    try:
        return load_preloaded_attack_traces(config.preloaded_attacks)
    except Exception:
        recorder.logger.warning("could not load the recorded attack traces; replaying without them", exc_info=True)
        return {}


async def _trace_anchor(provider: TraceProvider, recorder: RunRecorder) -> datetime | None:
    """The trace clock before the attack stage, so the round's traces exclude earlier traffic."""
    try:
        return trace_clock(await asyncio.to_thread(provider.get_traces))
    except Exception:
        recorder.logger.warning("could not read traces before the attack stage", exc_info=True)
        return None


async def _collect_attack_traces(
    provider: TraceProvider,
    recorder: RunRecorder,
    store: RoundStore,
    attacks: list[AttackRecord],
    anchor: datetime | None,
) -> dict[str, AgentTrace]:
    """Match this round's live attack hits to their traces and record them for a later replay.

    Runs whenever the global ``traces`` flag is on and live attackers ran, never depending on component
    flags: the recording is what makes a replay with traces possible. A trace failure never fails the
    round.
    """
    try:
        window = traces_since(await asyncio.to_thread(provider.get_traces), anchor)
        recorded = collect_attack_traces(attacks, window)
    except Exception:
        recorder.logger.warning("could not collect attack traces; defenders run without them", exc_info=True)
        return {}
    path = store.write_round_json(ATTACK_TRACES_FILENAME, dump_recording(recorded))
    recorder.emit(EventType.ARTIFACT_WRITTEN, artifact=ATTACK_TRACES_FILENAME, path=str(path), count=len(recorded))
    return by_prompt(recorded)
