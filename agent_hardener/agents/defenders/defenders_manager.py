# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from agent_hardener.agents.defenders.extraction_cache import ExtractionCache
from agent_hardener.agents.defenders.openshell_defender_v2.errors import LintViolationError
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import dump_policy_yaml, load_policy_from_yaml
from agent_hardener.agents.defenders.openshell_defender_v2.policy.merge import merge_candidates
from agent_hardener.models import (
    CURRENT_POLICY_KEY,
    AgentConfig,
    AttackRecord,
    DefenderAnalysis,
    DefenderInput,
    DefenderOutput,
    DefendersManagerInput,
    DefendersManagerOutput,
    ValidationFeedback,
)

logger = logging.getLogger(__name__)


# The attacked tool is inferred from the attacker's own phrasing, which fails whenever an attack does
# not announce its target. Reading it from the victim's ATOF stream instead needs a per-request scope
# to attribute against, and no harness emits one without being patched into it — a cost the victim
# contract (one line: attach Relay's middleware) is not worth paying.
_ATTACKED_TOOL_RE = re.compile(r"Use the (\S+) tool (?:to|for)", re.IGNORECASE)


def _parse_attacked_tool(attack_prompt: str) -> str:
    match = _ATTACKED_TOOL_RE.search(attack_prompt)
    return match.group(1) if match else ""


def _record_response_text(record: dict[str, Any]) -> str:
    """The agent's transcript/response for an attack record.

    Live attacker records key this ``response``; preloaded/replayed garak hitlog records
    (``agent_hardener.agents.attackers.hits.normalize_hit_record`` passes them through unchanged)
    key the same thing ``output``, often as ``{"text": "...", ...}`` rather than a plain string.
    Without this fallback, every preloaded-hitlog attack feeds the extractor an empty transcript.
    """
    value = record.get("response")
    if not value:
        value = record.get("output")
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        text = value.get("text")
        if isinstance(text, str):
            return text
    return ""


def _is_relay_guardrail_result(
    result: DefenderOutput, relay_plugins_path: Path | None, target_relay_plugins: Path | None
) -> bool:
    return (
        result.resource_type == "relay_guardrail_component"
        and relay_plugins_path is not None
        and target_relay_plugins is not None
    )


def _build_policy_patches(
    result: DefenderOutput,
    relay_plugins_path: Path | None,
    target_relay_plugins: Path | None,
    victim_policy_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build the correct patch structure based on the defender result type.

    For relay_guardrail_component results (guardrails_defender_v2), the component writer has already
    written the extended guardrail set to relay_plugins_path (the active-state copy). We need a
    victim_workflow_candidate patch so the adapter copies it to the deploy target and restarts.

    For a defender that produced policy YAML, return an openshell_policy_candidate patch so the
    deploy stage can upload it. The trigger is "a policy was produced", not the resource type: the
    OpenShell defender reports what it *attacked* (network, filesystem, landlock, process), never
    "openshell_policy", so keying off resource_type made this branch unreachable and no generated
    policy ever reached a sandbox.

    Does NOT write to ``victim_policy_path`` itself — concurrent defenders each compute a
    candidate against the same pristine policy, so writing each one here would clobber the
    others; ``DefendersManager.run`` merges every candidate and writes the result exactly once,
    before this function is ever called.
    """
    if _is_relay_guardrail_result(result, relay_plugins_path, target_relay_plugins):
        return [
            {
                "type": "relay_plugins_candidate",
                # Absolute: the deploy stage hands this to `openshell sandbox upload`, which runs
                # with the *victim project's* cwd, not this process's. A run-dir-relative path
                # resolves against the wrong root there and the upload fails with "local path does
                # not exist" — after every guardrail has been generated, so the round looks
                # successful right up until nothing reaches the victim.
                "candidate_relay_plugins_path": str(Path(relay_plugins_path).resolve()),
                "target_relay_plugins_path": str(target_relay_plugins),
                "requires_recreate": True,
                "changed": True,
            }
        ]
    if result.new_policy_yaml and victim_policy_path is not None:
        return [
            {
                "type": "openshell_policy_candidate",
                "candidate_policy_path": str(victim_policy_path),
                "changed": True,
                # Network policies hot-reload via `openshell policy set`; filesystem, landlock and
                # process changes only take effect on a fresh sandbox.
                "requires_recreate": result.resource_type != "network",
            }
        ]
    return [{"new_policy_yaml": result.new_policy_yaml}] if result.new_policy_yaml else []


class DefendersManager:
    async def run(
        self,
        input_data: DefendersManagerInput,
        run_defender_cb: Callable[[AgentConfig, DefenderInput], Awaitable[DefenderOutput]],
        validation_feedback: dict[str, ValidationFeedback] | None = None,
    ) -> DefendersManagerOutput:
        """Evaluate attacks, route to specific defenders, and aggregate policy patches."""
        routed = self._route_attacks(
            attacks=input_data.attacks,
            _summaries=input_data.attacker_summaries,
            available=input_data.available_defenders,
        )

        if not routed:
            return DefendersManagerOutput(
                ok=False, error="Defenders manager failed to map any attacks to available defenders."
            )

        all_selected_by_id = {d.agent_id: d for _, defenders in routed for d in defenders}
        routed_agent_ids = list(all_selected_by_id.keys())
        benign_requests = [r.payload for r in input_data.benign_requests]

        enriched_context = {
            **input_data.context,
            "round_id": input_data.round_id,
            "defender_extraction_cache": ExtractionCache(),
        }
        if "agent_relay_plugins" not in enriched_context and input_data.target.agent_relay_plugins is not None:
            enriched_context["agent_relay_plugins"] = str(input_data.target.agent_relay_plugins)

        relay_plugins_str: str | None = enriched_context.get("agent_relay_plugins")
        relay_plugins_path = Path(relay_plugins_str) if relay_plugins_str else None
        policy_path_str: str | None = enriched_context.get("victim_policy_path")
        victim_policy_path = Path(policy_path_str) if policy_path_str else None

        tasks = []
        task_meta: list[tuple[AgentConfig, DefenderInput]] = []
        for attack, defenders in routed:
            records = attack.records if attack.records else [{}]
            for record in records:
                record_prompt = str(record.get("prompt", attack.summary))
                record_response = _record_response_text(record)
                attacked_tool = _parse_attacked_tool(record_prompt) or attack.metadata.get("attacked_tool", "")
                for defender in defenders:
                    if validation_feedback and defender.agent_id not in validation_feedback:
                        continue
                    inp = DefenderInput(
                        attack_prompt=record_prompt,
                        agent_response=record_response,
                        attacked_tool=attacked_tool,
                        benign_requests=benign_requests,
                        context={**enriched_context, **defender.config},
                        relay_plugins_path=relay_plugins_path,
                        feedback=validation_feedback.get(defender.agent_id) if validation_feedback else None,
                        relay_victim_spec=input_data.relay_victim_spec,
                    )
                    tasks.append(run_defender_cb(defender, inp))
                    task_meta.append((defender, inp))

        if not tasks:
            return DefendersManagerOutput(
                ok=False, error="No defenders matched the validation feedback targets — nothing to retry."
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        self._merge_and_write_policy(
            task_meta,
            results,
            enriched_context,
            relay_plugins_path,
            victim_policy_path,
            input_data.target.agent_relay_plugins,
        )

        analyses: list[DefenderAnalysis] = []
        errors: list[str] = []

        for (defender, _), result in zip(task_meta, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"Agent {defender.name} raised exception: {result}")
            elif not result.ok:
                errors.append(f"Agent {defender.name} failed: {result.error_message}")
            else:
                analyses.append(
                    DefenderAnalysis(
                        agent_id=defender.agent_id,
                        agent_name=defender.name,
                        ok=result.ok,
                        policy_patches=_build_policy_patches(
                            result, relay_plugins_path, input_data.target.agent_relay_plugins, victim_policy_path
                        ),
                        metadata={
                            "iteration_count": result.iteration_count,
                            "resource_type": result.resource_type,
                            # Link this analysis to the specific guardrail it injected + the tool it attacked,
                            # so build_mitigations can pair each custom_guardrail_N with its motivating attack.
                            "guardrail_name": result.guardrail_name,
                            "attacked_tool": _.attacked_tool,
                        },
                        attack_prompt=_.attack_prompt,
                    )
                )

        return DefendersManagerOutput(
            ok=len(analyses) > 0,
            summary=f"Routed to {len(all_selected_by_id)} defenders across {len(input_data.attacks)} attacks. Generated {len(analyses)} analyses.",
            error=" | ".join(errors) if errors else None,
            routed_agent_ids=routed_agent_ids,
            analyses=analyses,
        )

    @staticmethod
    def _merge_and_write_policy(
        task_meta: list[tuple[AgentConfig, DefenderInput]],
        results: list[DefenderOutput | BaseException],
        enriched_context: dict[str, Any],
        relay_plugins_path: Path | None,
        victim_policy_path: Path | None,
        target_relay_plugins: Path | None,
    ) -> None:
        """Merge every concurrent defender's candidate policy and write the result exactly once.

        Every routed defender computed its candidate against the same pristine ``current_policy``
        snapshot (seeded once into ``enriched_context`` before the gather above), so none of them
        saw another's edit — writing each one individually would clobber the others.
        """
        if victim_policy_path is None:
            return
        candidate_texts = [
            result.new_policy_yaml
            for (_defender, _inp), result in zip(task_meta, results, strict=True)
            if not isinstance(result, BaseException)
            and result.ok
            and result.new_policy_yaml
            and not _is_relay_guardrail_result(result, relay_plugins_path, target_relay_plugins)
        ]
        if not candidate_texts:
            return
        original = load_policy_from_yaml(str(enriched_context.get(CURRENT_POLICY_KEY, "")))
        candidates = [load_policy_from_yaml(text) for text in candidate_texts]
        try:
            merged = merge_candidates(original, candidates)
        except LintViolationError:
            logger.exception("policy candidate merge widened access; leaving %s untouched", victim_policy_path)
            return
        victim_policy_path.write_text(dump_policy_yaml(merged), encoding="utf-8")

    @staticmethod
    def _route_attacks(
        attacks: list[AttackRecord], _summaries: list[str], available: list[AgentConfig]
    ) -> list[tuple[AttackRecord, list[AgentConfig]]]:
        """Route every attack to every available defender.

        This is currently a simple, non-selective rule rather than attack-aware routing — it
        doesn't try to match an attack's vector to a defender's stated capabilities (``_summaries``
        is unused for this reason). That's safe because each defender is responsible for its own
        scope: e.g. openshell_defender_v2 only ever mitigates within
        network_policies/network_middlewares and abstains (``ok=False``,
        ``out_of_scope``/``no_feasible_node``) on anything else, rather than force an irrelevant
        patch. A more selective routing rule can replace this later without changing that contract.
        """
        return [(attack, list(available)) for attack in attacks] if available else []
