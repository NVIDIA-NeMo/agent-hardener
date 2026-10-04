# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One feasibility predicate per node: ``check_<node>(ctx) -> Witness | None``.

Each predicate is self-contained and produces its own witness (the separating value) as a side
effect of checking — that's what lets both nodes be pure formatters downstream. Both are
deterministic and LLM-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..extraction.environment import endpoint_is_protected
from ..nodes.add_deny_rule import AddDenyRuleWitness
from ..nodes.remove_endpoint import RemoveEndpointWitness
from ..policy.globs import prefix_to_glob
from ..policy.query import find_endpoint, find_entry_for
from .benign_index import trie_for

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..models import DefenderContext, Witness
    from ..policy.schema import Endpoint


def _locate(ctx: DefenderContext) -> tuple[str, int, Endpoint] | None:
    """Common lookup: the entry key, endpoint index, and endpoint for ``ctx.cut``."""
    if ctx.cut is None or ctx.cut.host is None:
        return None
    found = find_entry_for(ctx.policy, ctx.cut.host, ctx.cut.port, ctx.cut.binary)
    if found is None:
        return None
    entry_key, entry = found
    endpoint = find_endpoint(entry, ctx.cut.host, ctx.cut.port)
    if endpoint is None:
        return None
    return entry_key, entry.endpoints.index(endpoint), endpoint


def check_remove_endpoint(ctx: DefenderContext) -> Witness | None:
    located = _locate(ctx)
    if located is None or ctx.benign_index is None:
        return None
    entry_key, endpoint_index, endpoint = located
    key = (ctx.cut.host, ctx.cut.port)
    if endpoint_is_protected(ctx.protected_endpoints, ctx.cut.host, ctx.cut.port):
        return None
    touched = key in (
        ctx.benign_index.methods.keys()
        | ctx.benign_index.tries.keys()
        | ctx.benign_index.binaries.keys()
        | ctx.benign_index.semantic_fields.keys()
        | ctx.benign_index.endpoints
    )
    if touched:
        return None
    return RemoveEndpointWitness(
        entry_key=entry_key, endpoint_index=endpoint_index, host=endpoint.host, port=endpoint.port
    )


def _check_add_deny_rule_semantic(
    ctx: DefenderContext, entry_key: str, endpoint_index: int, endpoint
) -> Witness | None:
    if ctx.overlap is None:
        return None
    field = ctx.cut.gql_field or ctx.cut.mcp_tool
    if not field or ctx.overlap.semantic_field_collision:
        return None
    field_kind = {
        "graphql": "graphql_field",
        "mcp": "mcp_tool",
        "json-rpc": "jsonrpc_method",
    }[endpoint.protocol]
    return AddDenyRuleWitness(
        entry_key=entry_key,
        endpoint_index=endpoint_index,
        protocol=endpoint.protocol,
        field_kind=field_kind,
        field_value=field,
    )


def _check_add_deny_rule_rest(ctx: DefenderContext, entry_key: str, endpoint_index: int, endpoint) -> Witness | None:
    if ctx.cut is None or not ctx.cut.path:
        return None
    if not endpoint.protocol or endpoint.tls == "skip":
        return None
    if ctx.benign_index is None:
        return None
    trie = trie_for(ctx.benign_index, ctx.cut.host, ctx.cut.port)
    prefix = trie.shallowest_clean_prefix(ctx.cut.path)
    if prefix is None:
        return None
    return AddDenyRuleWitness(
        entry_key=entry_key,
        endpoint_index=endpoint_index,
        glob=prefix_to_glob(prefix),
        attack_methods={ctx.cut.method} if ctx.cut.method else set(),
    )


def check_add_deny_rule(ctx: DefenderContext) -> Witness | None:
    located = _locate(ctx)
    if located is None:
        return None
    entry_key, endpoint_index, endpoint = located
    if endpoint_is_protected(ctx.protected_endpoints, ctx.cut.host, ctx.cut.port):
        return None
    if endpoint.protocol in ("graphql", "mcp", "json-rpc"):
        return _check_add_deny_rule_semantic(ctx, entry_key, endpoint_index, endpoint)
    return _check_add_deny_rule_rest(ctx, entry_key, endpoint_index, endpoint)


FEASIBILITY_CHECKS: dict[str, Callable[[DefenderContext], Witness | None]] = {
    "remove_endpoint": check_remove_endpoint,
    "add_deny_rule": check_add_deny_rule,
}


def infeasibility_reason(node_name: str, ctx: DefenderContext) -> str:
    """A short human-readable reason ``node_name`` didn't fire, for the abstain rationale."""
    reasons = {
        "remove_endpoint": "attacked host:port not in policy, also touched by predicted benign traffic, or binary filter excluded all entries",
        "add_deny_rule": (
            "endpoint isn't graphql/mcp/json-rpc with a separable field, and no path prefix "
            "separates the attack path from all predicted benign paths"
        ),
    }
    return reasons.get(node_name, "not feasible")
