# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``PolicyDelta`` — the vocabulary of edits a mitigation node may emit — and how to apply one."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .schema import DenyRule, Policy

_PassthroughModel = ConfigDict(extra="forbid")


class AddDenyRule(BaseModel):
    model_config = _PassthroughModel
    op: Literal["add_deny_rule"] = "add_deny_rule"
    entry_key: str
    endpoint_index: int
    rule: DenyRule


class RemoveEndpoint(BaseModel):
    model_config = _PassthroughModel
    op: Literal["remove_endpoint"] = "remove_endpoint"
    entry_key: str
    endpoint_index: int


class RemoveEntry(BaseModel):
    model_config = _PassthroughModel
    op: Literal["remove_entry"] = "remove_entry"
    entry_key: str


DeltaOp = AddDenyRule | RemoveEndpoint | RemoveEntry

_ALL_OP_TYPES: tuple[type[BaseModel], ...] = (
    AddDenyRule,
    RemoveEndpoint,
    RemoveEntry,
)


class PolicyDelta(BaseModel):
    """One node's proposed edit, as a small ordered list of ops (usually exactly one)."""

    model_config = _PassthroughModel
    ops: list[DeltaOp] = Field(default_factory=list)


def apply_delta(policy: Policy, delta: PolicyDelta) -> Policy:
    """Return a new :class:`Policy` with every op in ``delta`` applied, in order.

    Never mutates ``policy`` in place — callers (notably ``lint.assert_contraction``) need the
    original for comparison.
    """
    result = policy.model_copy(deep=True)
    for op in delta.ops:
        _apply_one(result, op)
    return result


def _apply_one(policy: Policy, op: DeltaOp) -> None:
    if isinstance(op, AddDenyRule):
        entry = policy.network_policies[op.entry_key]
        endpoint = entry.endpoints[op.endpoint_index]
        endpoint.deny_rules = [*(endpoint.deny_rules or []), op.rule]
    elif isinstance(op, RemoveEndpoint):
        entry = policy.network_policies[op.entry_key]
        del entry.endpoints[op.endpoint_index]
    elif isinstance(op, RemoveEntry):
        del policy.network_policies[op.entry_key]
    else:  # pragma: no cover - exhaustive over DeltaOp
        raise TypeError(f"unknown delta op: {op!r}")


def describe_delta(delta: PolicyDelta) -> str:
    """One-line-per-op human-readable summary, for rationale/log text."""
    lines = []
    for op in delta.ops:
        if isinstance(op, AddDenyRule):
            lines.append(
                f"deny {op.rule.method or op.rule.tool or op.rule.operation_type} on {op.entry_key}[{op.endpoint_index}]"
            )
        elif isinstance(op, RemoveEndpoint):
            lines.append(f"remove endpoint {op.entry_key}[{op.endpoint_index}]")
        elif isinstance(op, RemoveEntry):
            lines.append(f"remove entry {op.entry_key}")
    return "; ".join(lines)
