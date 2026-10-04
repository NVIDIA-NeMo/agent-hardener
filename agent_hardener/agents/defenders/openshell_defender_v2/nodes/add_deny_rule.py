# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``add_deny_rule`` — deny a REST path glob or a semantic protocol field never seen in benign traffic.

Pure formatter: the witness already carries the separating value (either a path glob provably
clean of benign traffic, or a GraphQL field / MCP tool / JSON-RPC method not seen in benign
traffic). Deny rules take precedence over allow rules and over ``access``, so the base allow set
stays intact.
"""

from __future__ import annotations

from typing import Literal

from ..models import Candidate, DefenderContext, Witness
from ..policy.patch import AddDenyRule, PolicyDelta
from ..policy.schema import DenyRule
from .base import assert_allowed_ops, build_rationale, confidence_for, locate_target

NAME = "add_deny_rule"
RESOURCE_TYPE = "l7_deny_rule"
ALLOWED_OPS = frozenset({AddDenyRule})


class AddDenyRuleWitness(Witness):
    node: str = NAME
    entry_key: str
    endpoint_index: int
    # REST path carveout branch.
    glob: str | None = None
    attack_methods: set[str] = frozenset()  # informational only; see _choose_method_matcher
    # Semantic protocol field branch (graphql/mcp/json-rpc).
    protocol: Literal["graphql", "mcp", "json-rpc"] | None = None
    field_kind: Literal["graphql_field", "graphql_operation", "mcp_tool", "jsonrpc_method"] | None = None
    field_value: str | None = None
    operation_type: str | None = None


def _choose_method_matcher(ctx: DefenderContext, witness: AddDenyRuleWitness) -> str:
    """``"*"`` in the normal case: the trie guarantees the glob matches no benign path at all,

    so denying every method on it can't touch benign traffic and can't be bypassed by switching
    methods (e.g. POST -> PUT). Only pin to explicit methods if the caller widened the glob past
    what the trie itself certified (not done by this node; kept for forward-compat with an
    optional LLM-widened glob upstream).
    """
    return "*"


def _graphql_deny_rule(witness: AddDenyRuleWitness) -> DenyRule:
    return DenyRule(operation_type=witness.operation_type, fields=[witness.field_value])


def _mcp_deny_rule(witness: AddDenyRuleWitness) -> DenyRule:
    return DenyRule(method="tools/call", tool=witness.field_value)


def _jsonrpc_deny_rule(witness: AddDenyRuleWitness) -> DenyRule:
    # JSON-RPC deny rules take an exact method string; globs are rejected on this protocol.
    return DenyRule(method=witness.field_value)


def _batch_collateral_note(ctx: DefenderContext, witness: AddDenyRuleWitness) -> str | None:
    """Warn when denying this tool/method may also deny benign traffic sharing its batch.

    MCP/JSON-RPC batches are denied wholesale if any call in the batch is denied — if a
    predicted benign request's tuples show this same tool/method co-occurring with the target
    endpoint, warn and force low confidence.
    """
    if witness.protocol not in ("mcp", "json-rpc"):
        return None
    for t in ctx.benign_tuples:
        if witness.field_value in (t.host, t.mcp_tool):
            return (
                f"predicted benign traffic references '{witness.field_value}' — denying this "
                "tool/method may also deny benign batches that include it"
            )
    return None


def _synthesize_semantic(ctx: DefenderContext, witness: AddDenyRuleWitness) -> tuple[DenyRule, list[str], bool]:
    if witness.protocol == "graphql":
        rule = _graphql_deny_rule(witness)
    elif witness.protocol == "mcp":
        rule = _mcp_deny_rule(witness)
    else:
        rule = _jsonrpc_deny_rule(witness)
    notes = [f"deny {witness.field_kind} '{witness.field_value}' on {witness.protocol}"]
    collateral = _batch_collateral_note(ctx, witness)
    low_confidence = False
    if collateral:
        notes.append(collateral)
        low_confidence = True
    return rule, notes, low_confidence


def _synthesize_rest(ctx: DefenderContext, witness: AddDenyRuleWitness, target) -> tuple[DenyRule, list[str], bool]:
    method = _choose_method_matcher(ctx, witness)
    rule = DenyRule(method=method, path=witness.glob)
    notes = [f"deny {method} {witness.glob} — provably touches no predicted benign path"]
    if target.endpoint.options and target.endpoint.options.get("allow_encoded_slash"):
        notes.append("attack path may use encoded slashes; allow_encoded_slash left untouched deliberately")
    return rule, notes, False


def synthesize(ctx: DefenderContext, witness: Witness) -> Candidate:
    assert isinstance(witness, AddDenyRuleWitness)
    target = locate_target(ctx, witness.entry_key, witness.endpoint_index)

    if witness.protocol is not None:
        rule, notes, force_low = _synthesize_semantic(ctx, witness)
    else:
        rule, notes, force_low = _synthesize_rest(ctx, witness, target)

    delta = PolicyDelta(
        ops=[AddDenyRule(entry_key=witness.entry_key, endpoint_index=witness.endpoint_index, rule=rule)]
    )
    assert_allowed_ops(_Node(), delta)

    confidence = "low" if force_low else confidence_for(ctx, target.endpoint.host, target.endpoint.port)

    return Candidate(
        node=NAME,
        delta=delta.model_dump(mode="json"),
        resource_type=RESOURCE_TYPE,
        confidence=confidence,
        rationale=build_rationale(NAME, target, notes),
    )


class _Node:
    name = NAME
    resource_type = RESOURCE_TYPE
    allowed_ops = ALLOWED_OPS
    synthesize = staticmethod(synthesize)


NODE = _Node()
