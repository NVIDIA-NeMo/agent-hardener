# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-role agent I/O contracts and the standard run input/output shapes.

Attacker, victim, and validator wrappers all conform to ``run(request, agent) -> AgentRunOutput``.
Defenders deliberately diverge: they take :class:`DefenderInput` and return :class:`DefenderOutput`
and are fanned in by a manager, because they mutate the shared victim policy and need coordinated
routing. See ``agent_hardener/runtime/stages/defense.py`` for that boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from agent_hardener.models.agent import AgentConfig, TargetInput
from agent_hardener.models.base import AgentHardenerModel, ValidatorKind
from agent_hardener.models.infra import RelayVictimSpec


class Artifact(AgentHardenerModel):
    """A typed file reference that agents declare in their output for the display layer."""

    type: str = Field(min_length=1)
    path: Path
    label: str | None = None


class BenignRequest(AgentHardenerModel):
    """Represents a baseline, normal user request for the defenders to protect."""

    payload: str


class AttackRecord(AgentHardenerModel):
    """Normalized attacker output."""

    agent_id: str
    agent_name: str
    ok: bool = True
    summary: str = ""
    records: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None


class ValidationFeedback(AgentHardenerModel):
    """Structured failure details from a previous iteration's validators, fed back to defenders."""

    false_negatives: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)
    previous_policy_yaml: str | None = None


class DefenderInput(AgentHardenerModel):
    """Standard input passed from the manager to every individual defender."""

    attack_prompt: str
    agent_response: str
    attacked_tool: str
    benign_requests: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    #: The run's active-state NeMo Relay plugin config; the guardrails defender extends it.
    relay_plugins_path: Path | None = None
    feedback: ValidationFeedback | None = None
    # The run's victim_control.config["relay_victim"] spec (agent_env/egress/backends), so extraction
    # can treat the agent's own required endpoints as ground-truth benign.
    relay_victim_spec: RelayVictimSpec | None = None


class DefenderOutput(AgentHardenerModel):
    """Standard output returned from every individual defender to the manager."""

    ok: bool
    error_message: str | None = None
    new_policy_yaml: str | None = None
    iteration_count: int = 0
    resource_type: str | None = None
    # For the guardrails defender: the ``custom_guardrail_N`` guardrail it generated, so the manager can
    # link this specific guardrail back to the attack that motivated it (see build_mitigations).
    guardrail_name: str | None = None


class DefendersManagerInput(AgentHardenerModel):
    """Specific input required by the Defenders Manager to make routing decisions."""

    round_id: str
    target: TargetInput
    context: dict[str, Any] = Field(default_factory=dict)
    attacks: list[AttackRecord] = Field(default_factory=list)
    attacker_summaries: list[str] = Field(default_factory=list)
    benign_requests: list[BenignRequest] = Field(default_factory=list)
    available_defenders: list[AgentConfig] = Field(default_factory=list)
    # The run's victim_control.config["relay_victim"] spec, threaded into every DefenderInput built below.
    relay_victim_spec: RelayVictimSpec | None = None


class DefenderAnalysis(AgentHardenerModel):
    """Normalized defender output with policy patch suggestions."""

    agent_id: str
    agent_name: str
    ok: bool = True
    summary: str = ""
    policy_patches: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None
    attack_prompt: str | None = None


class DefendersManagerOutput(AgentHardenerModel):
    """Consolidated output from the routing and execution of defenders."""

    ok: bool = True
    summary: str = ""
    error: str | None = None
    routed_agent_ids: list[str] = Field(default_factory=list)
    analyses: list[DefenderAnalysis] = Field(default_factory=list)


class VictimResult(AgentHardenerModel):
    """Normalized victim run output."""

    agent_id: str
    agent_name: str
    ok: bool = True
    summary: str = ""
    observations: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None


class ValidatorReport(AgentHardenerModel):
    """Normalized attack or benign validation output."""

    agent_id: str
    agent_name: str
    kind: ValidatorKind
    ok: bool = True
    summary: str = ""
    findings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    error: str | None = None
    false_negatives: list[str] = Field(default_factory=list)
    false_positives: list[str] = Field(default_factory=list)


class VictimControlResult(AgentHardenerModel):
    """Policy-control adapter output."""

    ok: bool = True
    summary: str = ""
    policy_patches: list[dict[str, Any]] = Field(default_factory=list)
    redeploy_intent: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


# Well-known ``AgentRunInput.context`` key: the ready-to-use directory an agent writes its artifacts into,
# resolved by the authority and injected as a string (agents cross a serializable boundary, so no Path).
ARTIFACT_DIR_KEY = "artifact_dir"

# Well-known context key carrying the current victim policy YAML (text) the policy defender edits — seeded
# from the run's initial (inferred) policy so the defender starts from it rather than an empty document.
CURRENT_POLICY_KEY = "current_policy"


class AgentRunInput(AgentHardenerModel):
    """Standard POST /run input for all wrapper services."""

    round_id: str = Field(min_length=1)
    target: TargetInput
    context: dict[str, Any] = Field(default_factory=dict)
    iteration: int = Field(default=1, ge=1)
    attacks: list[AttackRecord] = Field(default_factory=list)
    attacker_summaries: list[str] = Field(default_factory=list)
    benign_requests: list[BenignRequest] = Field(default_factory=list)
    defender_analyses: list[DefenderAnalysis] = Field(default_factory=list)
    policy_patches: list[dict[str, Any]] = Field(default_factory=list)
    victim_result: VictimResult | None = None
    validator_kind: ValidatorKind | None = None


# Union of every normalized agent output; the concrete type is chosen by the agent's role.
AgentRunOutput = AttackRecord | DefenderAnalysis | VictimResult | ValidatorReport
