# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compute the per-axis collision between the attack's cut tuple and the benign index.

Tuned for recall on collisions per the spec: a missed collision ships a confident bad patch
(the worst outcome — it over-blocks silently); a spurious collision only costs the selector a
fallback rung. So every check here is deliberately permissive about what counts as "colliding".
"""

from __future__ import annotations

from ..models import OverlapVector, RequestTuple
from .benign_index import (
    BenignIndex,
    binaries_for,
    methods_for,
    queries_for,
    semantic_fields_for,
    sources_for,
    trie_for,
)


def compute_overlap(cut: RequestTuple, index: BenignIndex) -> OverlapVector:
    """The collision vector for ``cut`` against everything the benign index knows about its

    ``(host, port)`` endpoint.
    """
    key_hosts = {h for (h, _p) in index.methods.keys() | index.tries.keys() | index.binaries.keys() | index.endpoints}
    host_collision = cut.host is not None and cut.host in key_hosts
    port_collision = (cut.host, cut.port) in (
        index.methods.keys()
        | index.tries.keys()
        | index.binaries.keys()
        | index.semantic_fields.keys()
        | index.endpoints
    )

    methods = methods_for(index, cut.host, cut.port)
    binaries = binaries_for(index, cut.host, cut.port)
    trie = trie_for(index, cut.host, cut.port)
    semantic_fields = semantic_fields_for(index, cut.host, cut.port)
    queries = queries_for(index, cut.host, cut.port)
    sources = sources_for(index, cut.host, cut.port)

    method_collision = bool(cut.method) and cut.method.upper() in methods
    binary_collision = bool(cut.binary) and cut.binary in binaries
    path_collision = bool(cut.path) and "benign" in trie.subtree_owners(cut.path)
    query_collision = bool(cut.query) and cut.query in queries
    semantic_field = cut.gql_field or cut.mcp_tool
    semantic_field_collision = bool(semantic_field) and semantic_field in semantic_fields

    evidence: dict[str, list[str]] = {}
    for axis, hit in (
        ("host", host_collision),
        ("port", port_collision),
        ("method", method_collision),
        ("binary", binary_collision),
        ("path", path_collision),
        ("query", query_collision),
        ("semantic_field", semantic_field_collision),
    ):
        if hit:
            evidence[axis] = sources

    return OverlapVector(
        host_collision=host_collision,
        port_collision=port_collision,
        binary_collision=binary_collision,
        method_collision=method_collision,
        path_collision=path_collision,
        query_collision=query_collision,
        semantic_field_collision=semantic_field_collision,
        evidence=evidence,
    )
