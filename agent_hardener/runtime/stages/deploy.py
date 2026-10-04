# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The deploy stage: push the defenders' changes to the victim and snapshot the deployed files.

This is the one non-agent stage — infra dispatch rather than an LLM call — so it composes the two
uploaders instead of an :class:`AgentInvoker`.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_hardener.events import EventType
from agent_hardener.models import VictimControlResult
from agent_hardener.runtime.adapters import openshell_policy_patch, relay_guardrail_patch
from agent_hardener.swarm_tracker import victim_active_state_dir

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import SessionConfig
    from agent_hardener.runtime.adapters import Uploader
    from agent_hardener.runtime.round_store import RoundStore
    from agent_hardener.runtime.stages.context import DefenseResult, IterationContext


class DeployStage:
    """Deploy the defenders' changes to the victim, one named step per change type."""

    def __init__(self, config: SessionConfig, openshell: Uploader, relay: Uploader) -> None:
        self.config = config
        self.openshell = openshell
        self.relay = relay

    async def run(self, ctx: IterationContext, defense: DefenseResult) -> VictimControlResult:
        recorder = ctx.recorder
        policy_patches = defense.policy_patches
        victim_control_start = perf_counter()
        recorder.emit(
            EventType.VICTIM_CONTROL_STARTED,
            iteration=ctx.iteration,
            patch_count=len(policy_patches),
            defender_count=len(defense.analyses),
        )

        results: list[dict[str, Any]] = []
        ok = True

        policy = openshell_policy_patch(policy_patches)
        if policy:
            recreate = bool(policy.get("requires_recreate", False))
            recorder.emit(
                EventType.OPENSHELL_UPLOAD,
                iteration=ctx.iteration,
                candidate_policy_path=policy["candidate_policy_path"],
                recreate=recreate,
            )
            result = await self.openshell.upload(policy["candidate_policy_path"], recreate=recreate)
            ok = ok and result.ok
            results.append(
                {
                    "step": "openshell",
                    "candidate_policy_path": policy["candidate_policy_path"],
                    "recreate": recreate,
                    "ok": result.ok,
                    "output": result.output,
                }
            )

        guardrails = relay_guardrail_patch(policy_patches)
        if guardrails:
            recorder.emit(
                EventType.RELAY_POLICY_UPLOAD,
                iteration=ctx.iteration,
                target_relay_plugins_path=guardrails["target_relay_plugins_path"],
            )
            result = await self.relay.upload(guardrails["candidate_relay_plugins_path"])
            ok = ok and result.ok
            results.append(
                {
                    "step": "relay",
                    "target_relay_plugins_path": guardrails["target_relay_plugins_path"],
                    "ok": result.ok,
                    "output": result.output,
                }
            )

        steps = [entry["step"] for entry in results]
        summary = f"deployed: {', '.join(steps)}" if steps else "no defender changes to deploy"
        victim_control = VictimControlResult(
            ok=ok,
            summary=summary,
            policy_patches=policy_patches,
            redeploy_intent=bool(results),
            metadata={
                "round_id": recorder.round_id,
                "defender_count": len(defense.analyses),
                "adapter": "uploaders",
                "results": results,
            },
            error=None if ok else summary,
        )
        victim_control_duration = perf_counter() - victim_control_start
        recorder.emit(
            EventType.VICTIM_CONTROL_COMPLETED,
            iteration=ctx.iteration,
            ok=victim_control.ok,
            duration_seconds=round(victim_control_duration, 6),
        )
        if ctx.iteration_dir is not None:
            ctx.store.write_component(ctx.iteration_dir, "victim_control", "result", victim_control)
            self._snapshot_deployed_victim(ctx.store, ctx.iteration_dir, ctx.run_dir)
        return victim_control

    def _snapshot_deployed_victim(self, store: RoundStore, iteration_dir: Path, run_dir: Path | None) -> None:
        """Snapshot the victim agent files as deployed for this iteration (post-defender mutation)."""
        storage = self.config.storage
        active_state_dir = victim_active_state_dir(run_dir) if run_dir is not None else None
        policy_src = (
            active_state_dir / storage.victim_policy_path.name
            if (active_state_dir and storage.victim_policy_path)
            else storage.victim_policy_path
        )
        workflow_src = (
            active_state_dir / storage.victim_relay_plugins_path.name
            if (active_state_dir and storage.victim_relay_plugins_path)
            else storage.victim_relay_plugins_path
        )
        store.snapshot_victim(iteration_dir, policy_src, workflow_src)
