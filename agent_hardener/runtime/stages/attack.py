# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attack stage: load preloaded attacks and fan out over the configured attackers."""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.attackers.preloaded import load_preloaded_attacks
from agent_hardener.concurrency import gather_limited_ordered
from agent_hardener.errors import AttackerError
from agent_hardener.events import EventType
from agent_hardener.models import ARTIFACT_DIR_KEY, AgentRunInput, AttackRecord
from agent_hardener.runtime.stages.agent_invoker import all_ok, collect_outputs
from agent_hardener.storage import to_jsonable
from agent_hardener.swarm_tracker import GARAK

if TYPE_CHECKING:
    from agent_hardener.models import SessionConfig
    from agent_hardener.runtime.round_store import RoundStore
    from agent_hardener.runtime.run_recorder import RunRecorder
    from agent_hardener.runtime.stages.agent_invoker import AgentInvoker


# Per-attacker cap on streamed transcript rows — garak can produce many hits; keep the live feed usable.
_MAX_EXCHANGES_PER_AGENT = 100


def _hit_text(value: Any) -> str:
    """Flatten a garak hit field (str / list / nested) to display text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(_hit_text(item) for item in value)
    return str(value)


def _emit_attack_exchanges(recorder: RunRecorder, outputs: list[AttackRecord]) -> None:
    """Stream one AGENT_EXCHANGE per attacker hit so the UI can show each agent's prompt<->response transcript."""
    for attack in outputs:
        for record in attack.records[:_MAX_EXCHANGES_PER_AGENT]:
            request = _hit_text(record.get("prompt") or record.get("attack_prompt") or record.get("command"))
            response = _hit_text(record.get("output") or record.get("victim_response") or record.get("result"))
            if not (request or response):
                continue
            recorder.emit(
                EventType.AGENT_EXCHANGE,
                agent_id=attack.agent_id,
                agent_name=attack.agent_name,
                agent_role="attacker",
                phase="attackers",
                request=request,
                response=response,
                label=str(record.get("detector") or record.get("probe") or ""),
                ok=bool(attack.ok),
            )


class AttackStage:
    """Run preloaded attacks plus a fan-out over ``config.attackers`` into attack records.

    Attackers run once per round, so all their artifacts are round-scoped: ``attacks.json``, the
    per-attacker component files, and the garak scan dir all live under ``round_<N>/``.
    """

    def __init__(self, config: SessionConfig, invoker: AgentInvoker) -> None:
        self.config = config
        self.invoker = invoker

    async def run(self, recorder: RunRecorder, store: RoundStore) -> list[AttackRecord]:
        preloaded_attacks = load_preloaded_attacks(self.config.preloaded_attacks)
        if preloaded_attacks:
            recorder.logger.info("loaded preloaded attacks count=%s", len(preloaded_attacks))
            recorder.emit(
                EventType.PRELOADED_ATTACKS_LOADED,
                count=len(preloaded_attacks),
                record_count=sum(len(attack.records) for attack in preloaded_attacks),
            )
        # Hand file-producing attackers (e.g. the garak agent breaker) their ready round-scoped output dir.
        request = AgentRunInput(
            round_id=recorder.round_id,
            target=self.config.target,
            context={**self.invoker.request_context(), ARTIFACT_DIR_KEY: str(store.round_artifact_dir(GARAK))},
        )
        phase_started_at = perf_counter()
        recorder.emit(EventType.PHASE_STARTED, phase="attackers", count=len(self.config.attackers))
        results = await gather_limited_ordered(
            self.config.attackers,
            len(self.config.attackers) or 1,
            lambda agent: self.invoker.run_agent(
                agent, request, "attackers", recorder.mission_id, recorder.logger, recorder.events
            ),
            return_exceptions=True,
        )
        outputs = [
            *preloaded_attacks,
            *collect_outputs(results, self.config.attackers, request, AttackRecord, recorder.logger),
        ]
        phase_duration = perf_counter() - phase_started_at
        recorder.emit(
            EventType.PHASE_COMPLETED,
            phase="attackers",
            count=len(outputs),
            ok=all_ok(outputs),
            duration_seconds=round(phase_duration, 6),
        )

        attacks_path = store.write_round_json("attacks.json", outputs)
        recorder.emit(
            EventType.ARTIFACT_WRITTEN,
            artifact="attacks.json",
            path=str(attacks_path),
            count=len(outputs),
        )
        recorder.emit(EventType.ATTACKERS_COMPLETED, count=len(outputs))
        # Serialized records so a live observer can render the Garak summary table mid-run.
        recorder.emit(EventType.ATTACK_SUMMARY, attacks=[to_jsonable(attack) for attack in outputs])
        # Per-hit transcript rows keyed to each attacker so the UI can show its prompt<->response exchanges.
        _emit_attack_exchanges(recorder, outputs)

        # Attackers run once for the round; their component files live at the round dir.
        live_outputs = outputs[len(outputs) - len(self.config.attackers) :]
        for agent, output in zip(self.config.attackers, live_outputs, strict=False):
            store.write_component(store.round_dir, "attacker", agent.name, output)

        # A live attacker that failed (e.g. timed out) leaves no valid attack corpus — surface it as a
        # hard error rather than a spurious "0 hits". Artifacts above are already written for post-mortem.
        raise_on_attacker_failure(self.config.attackers, live_outputs)
        return outputs


def raise_on_attacker_failure(attackers: list[Any], live_outputs: list[AttackRecord]) -> None:
    """Raise :class:`AttackerError` if any live attacker did not complete (``ok=False``).

    A failed/partial attacker produces no scored hits, so a run that continued would report "0 hits" —
    a false negative that reads as "the victim resisted" when the scan never finished. Replay/preloaded
    runs empty ``config.attackers``, so this is a no-op for them.
    """
    failed = [(agent, out) for agent, out in zip(attackers, live_outputs, strict=False) if not out.ok]
    if not failed:
        return
    names = ", ".join(agent.name for agent, _ in failed)
    reason = "; ".join(str(out.error or "did not complete") for _, out in failed)
    raise AttackerError(f"attacker(s) {names} did not complete: {reason}")
