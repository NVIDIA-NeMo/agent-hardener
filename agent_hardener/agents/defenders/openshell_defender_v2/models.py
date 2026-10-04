# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal typed data model for the deterministic feasibility/selector pipeline.

Everything here except :class:`DefenderContext` crosses between deterministic functions and
LLM-boundary functions, so it is a strict :class:`AgentHardenerModel`. :class:`DefenderContext` never
serializes (it's the mutable working state passed between the private helpers in
``openshell_defender_v2_agent.py``), so it's a plain dataclass instead — that also sidesteps a
circular import between this module and ``policy.schema`` / ``analysis.benign_index``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic import Field

from agent_hardener.models.base import AgentHardenerModel

if TYPE_CHECKING:
    from agent_hardener.models.contracts import DefenderInput

    from .analysis.benign_index import BenignIndex
    from .extraction.environment import EndpointKey
    from .feedback import FeedbackDirective
    from .policy.schema import Policy

AbstainCode = Literal["out_of_scope", "unobservable", "no_feasible_node", "extraction_failed", "protected_endpoint"]


class RequestTuple(AgentHardenerModel):
    """The axes an OpenShell network rule can actually see and match on."""

    binary: str | None = None
    host: str | None = None
    port: int | None = None
    protocol: str | None = None
    method: str | None = None
    path: str | None = None
    query: str | None = None
    gql_field: str | None = None
    mcp_tool: str | None = None


class BenignPrediction(AgentHardenerModel):
    """Structural (never benign/attack) extraction of the tuples a known-benign request hits.

    ``source_request`` is one of the already-ground-truth-benign strings from
    ``DefenderInput.benign_requests``. This model only records *what it would hit*, never
    *whether it's benign* — that's already decided upstream.
    """

    source_request: str
    tuples: list[RequestTuple] = Field(default_factory=list)
    confidence: float = 1.0


class HarmCertificate(AgentHardenerModel):
    """Why the attack succeeded and which single request in the chain to cut."""

    harm_class: str
    channel: str
    cut_index: int
    reasoning: str = ""


class OverlapVector(AgentHardenerModel):
    """Per-axis collision between the attack's cut tuple and the benign index.

    Tuned for recall on collisions: a missed collision ships a confident bad patch, a
    spurious one only costs a fallback rung in the selector.
    """

    host_collision: bool = False
    port_collision: bool = False
    binary_collision: bool = False
    method_collision: bool = False
    path_collision: bool = False
    query_collision: bool = False
    semantic_field_collision: bool = False
    evidence: dict[str, list[str]] = Field(default_factory=dict)


class Witness(AgentHardenerModel):
    """Discriminated-union base for the separating value a feasibility check produces.

    Each node defines its own subclass (see ``nodes/*.py``) carrying whatever fields it needs
    to become a formatter; ``node`` is the discriminator.
    """

    node: str


class Selection(AgentHardenerModel):
    """Output of ``analysis.selector.select`` — the head to synthesize plus the rest of the chain."""

    chosen: str | None = None
    witness: Witness | None = None
    fallbacks: list[tuple[str, Witness]] = Field(default_factory=list)
    infeasible: dict[str, str] = Field(default_factory=dict)


class Candidate(AgentHardenerModel):
    """A synthesized, not-yet-linted policy delta from exactly one node."""

    node: str
    delta: dict = Field(default_factory=dict)  # policy.patch.PolicyDelta, dumped to avoid an import cycle
    resource_type: str
    guardrail_name: str | None = None
    confidence: Literal["high", "low"] = "high"
    rationale: str = ""


class AbstainReason(AgentHardenerModel):
    """Why no patch was produced, in the vocabulary the manager/reviewer expects."""

    code: AbstainCode
    detail: str = ""
    per_node_reasons: dict[str, str] = Field(default_factory=dict)


@dataclass
class DefenderContext:
    """Mutable working state threaded through ``_build_context`` -> ``_resolve_selection`` ->

    ``_synthesize``. Never crosses a serialization boundary, so it's a plain dataclass.
    """

    defender_input: DefenderInput
    policy: Policy
    attack_tuples: list[RequestTuple] = field(default_factory=list)
    cut: RequestTuple | None = None
    harm_cert: HarmCertificate | None = None
    benign_index: BenignIndex | None = None
    # Flattened predicted (+ feedback-counterexample) benign tuples, kept alongside the aggregated
    # benign_index because add_deny_rule's batch-collateral check needs paired per-request facts
    # that the index's per-axis sets alone don't preserve.
    benign_tuples: list[RequestTuple] = field(default_factory=list)
    benign_predictions: list[BenignPrediction] = field(default_factory=list)
    overlap: OverlapVector | None = None
    feedback: FeedbackDirective | None = None
    iteration: int = 0
    # Endpoints (backends[].allowlist + the LLM's own inference endpoint) no mitigation node may
    # ever touch — an absolute block-list, distinct in kind from benign_index/benign_tuples (a
    # "known benign traffic" signal, not a guarantee nothing may ever act on that endpoint).
    protected_endpoints: set[EndpointKey] = field(default_factory=set)
