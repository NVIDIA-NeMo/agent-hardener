# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared contract, delta-op vocabulary, and helpers for the two mitigation nodes.

Neither node calls an LLM — the witness a feasibility check hands them already carries the
separating value, so ``synthesize`` is a pure formatter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

from ..config import load_config
from ..extraction.benign import coverage_for_endpoint

if TYPE_CHECKING:
    from ..models import Candidate, DefenderContext, Witness
    from ..policy.patch import PolicyDelta
    from ..policy.schema import Endpoint, NetworkPolicyEntry


class MitigationNode(Protocol):
    name: str
    resource_type: str
    allowed_ops: frozenset[type]

    def synthesize(self, ctx: DefenderContext, witness: Witness) -> Candidate: ...


NODE_REGISTRY: dict[str, MitigationNode] = {}
RESOURCE_TYPES: dict[str, str] = {}


def register(node: MitigationNode) -> None:
    NODE_REGISTRY[node.name] = node
    RESOURCE_TYPES[node.name] = node.resource_type


def get_node(name: str) -> MitigationNode:
    return NODE_REGISTRY[name]


@dataclass
class TargetRef:
    entry_key: str
    entry: NetworkPolicyEntry
    endpoint_index: int
    endpoint: Endpoint


def locate_target(ctx: DefenderContext, entry_key: str, endpoint_index: int) -> TargetRef:
    """Resolve a witness's ``(entry_key, endpoint_index)`` pair against ``ctx.policy``."""
    entry = ctx.policy.network_policies[entry_key]
    return TargetRef(
        entry_key=entry_key,
        entry=entry,
        endpoint_index=endpoint_index,
        endpoint=entry.endpoints[endpoint_index],
    )


def confidence_for(ctx: DefenderContext, host: str | None, port: int | None) -> Literal["high", "low"]:
    """Degrade to ``"low"`` confidence when the predicted-benign corpus is thin.

    ``"low"`` when fewer than ``config.min_corpus_coverage`` predicted benign requests touch this
    endpoint — the separation this node is about to draw is barely evidenced.
    """
    config = load_config()
    count = coverage_for_endpoint(ctx.benign_predictions, host, port)
    return "low" if count < config.min_corpus_coverage else "high"


def build_rationale(node: str, target: TargetRef, notes: list[str]) -> str:
    """Human-reviewer-facing summary: what changed, where, and why (including caveats)."""
    header = f"[{node}] {target.entry_key}[{target.endpoint_index}] ({target.endpoint.host}:{target.endpoint.port})"
    if not notes:
        return header
    return header + " — " + "; ".join(notes)


def assert_allowed_ops(node: MitigationNode, delta: PolicyDelta) -> None:
    """Node-local lint: raise if ``delta`` contains an op type outside ``node.allowed_ops``.

    Structural and cheap, and it catches a bug before the patch ever leaves ``nodes/`` —
    ``policy.lint.assert_contraction`` still runs in the agent afterward as defense in depth,
    not a replacement for this.
    """
    for op in delta.ops:
        if type(op) not in node.allowed_ops:
            raise TypeError(f"{node.name} emitted disallowed op {type(op).__name__}")
