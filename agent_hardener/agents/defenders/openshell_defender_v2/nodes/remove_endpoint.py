# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``remove_endpoint`` — drop an endpoint (or entry) benign traffic never touches.

Guarded by requiring the host:port to be entirely absent from predicted benign traffic.
"""

from __future__ import annotations

from ..models import Candidate, DefenderContext, Witness
from ..policy.patch import PolicyDelta, RemoveEndpoint, RemoveEntry
from .base import assert_allowed_ops, build_rationale, locate_target

NAME = "remove_endpoint"
RESOURCE_TYPE = "endpoint_removal"
ALLOWED_OPS = frozenset({RemoveEndpoint, RemoveEntry})

# Well-known DefenderInput.context key an upstream Policy Advisor may set identifying which
# advisor-approved rule this endpoint corresponds to, for a friendlier revert-flavored rationale.
POLICY_ADVISOR_CHUNK_KEY = "policy_advisor_chunk_id"


class RemoveEndpointWitness(Witness):
    node: str = NAME
    entry_key: str
    endpoint_index: int
    host: str
    port: int | None


def _is_last_endpoint(entry) -> bool:
    return len(entry.endpoints) <= 1


def _originating_proposal(ctx: DefenderContext) -> str | None:
    value = ctx.defender_input.context.get(POLICY_ADVISOR_CHUNK_KEY)
    return str(value) if value else None


def synthesize(ctx: DefenderContext, witness: Witness) -> Candidate:
    assert isinstance(witness, RemoveEndpointWitness)
    target = locate_target(ctx, witness.entry_key, witness.endpoint_index)
    entry = target.entry

    # Guard: don't collaterally remove a benign endpoint served by the same entry through a
    # different index when RemoveEntry would be used instead of RemoveEndpoint.
    if _is_last_endpoint(entry):
        ops = [RemoveEntry(entry_key=witness.entry_key)]
    else:
        other_hosts_benign = any(i != witness.endpoint_index for i, ep in enumerate(entry.endpoints))
        if not other_hosts_benign:
            raise ValueError("remove_endpoint: no other endpoint on this entry to preserve")
        ops = [RemoveEndpoint(entry_key=witness.entry_key, endpoint_index=witness.endpoint_index)]

    delta = PolicyDelta(ops=ops)
    assert_allowed_ops(_Node(), delta)

    proposal = _originating_proposal(ctx)
    notes = [f"host:port {witness.host}:{witness.port} entirely absent from predicted benign traffic"]
    if proposal:
        notes.append(f"reverts policy-advisor-approved rule {proposal}")

    return Candidate(
        node=NAME,
        delta=delta.model_dump(mode="json"),
        resource_type=RESOURCE_TYPE,
        confidence="high",
        rationale=build_rationale(NAME, target, notes),
    )


class _Node:
    name = NAME
    resource_type = RESOURCE_TYPE
    allowed_ops = ALLOWED_OPS
    synthesize = staticmethod(synthesize)


NODE = _Node()
