# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Inverted per-``(host, port)`` indexes over predicted benign request tuples.

Built once per ``run()`` call from ``extraction.benign.predict_benign_tuples`` output, then
consulted by ``analysis.overlap`` and every ``analysis.feasibility.check_*`` predicate. Not a
:class:`pydantic.BaseModel` — it holds a :class:`~.trie.PathTrie` per endpoint, which isn't a
plain-data type worth modeling strictly, and this object never crosses a serialization boundary.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .trie import PathTrie

if TYPE_CHECKING:
    from ..models import BenignPrediction, RequestTuple

EndpointKey = tuple[str | None, int | None]


class BenignIndex:
    def __init__(self) -> None:
        self.methods: dict[EndpointKey, set[str]] = defaultdict(set)
        self.binaries: dict[EndpointKey, set[str]] = defaultdict(set)
        self.tries: dict[EndpointKey, PathTrie] = {}
        self.semantic_fields: dict[EndpointKey, set[str]] = defaultdict(set)
        self.queries: dict[EndpointKey, set[str]] = defaultdict(set)
        self.sources: dict[EndpointKey, list[str]] = defaultdict(list)
        # Every (host, port) any absorbed tuple ever touched, regardless of which (if any) other
        # axis it carried. A bare host/port tuple (e.g. a relay_victim backend allowlist entry with
        # no method/path/binary/semantic field) still marks the endpoint as known-benign; without
        # this, such a tuple leaves no trace in any of the field-specific dicts above and a
        # feasibility check consulting only those sees "never touched" and removes the endpoint.
        self.endpoints: set[EndpointKey] = set()

    def _absorb(self, tuples: list[RequestTuple], source: str | None = None) -> None:
        for t in tuples:
            key = (t.host, t.port)
            if t.host is not None:
                self.endpoints.add(key)
            if t.method:
                self.methods[key].add(t.method.upper())
            if t.binary:
                self.binaries[key].add(t.binary)
            if t.path:
                self.tries.setdefault(key, PathTrie()).insert(t.path, "benign")
            semantic = t.gql_field or t.mcp_tool
            if semantic:
                self.semantic_fields[key].add(semantic)
            if t.query:
                self.queries[key].add(t.query)
            if source is not None:
                self.sources[key].append(source)


def build_index(predictions: list[BenignPrediction]) -> BenignIndex:
    """Fold every predicted benign request's tuples into one queryable index."""
    index = BenignIndex()
    for prediction in predictions:
        index._absorb(prediction.tuples, source=prediction.source_request)
    return index


def merge_counterexamples(index: BenignIndex, tuples: list[RequestTuple]) -> BenignIndex:
    """Fold validator-confirmed false-positive tuples (from ``feedback``) into ``index`` in place.

    These are strictly more authoritative than predictions — a validator actually observed one
    of these get wrongly blocked — so they're absorbed the same way but without prediction
    uncertainty. Must run before ``overlap.compute_overlap`` so round 2+ never repeats a mistake
    the previous round already proved.
    """
    index._absorb(tuples, source="<validator-counterexample>")
    return index


def has_endpoint(index: BenignIndex, host: str | None, port: int | None) -> bool:
    """True if any absorbed tuple touched ``(host, port)``, on any axis (including none)."""
    return (host, port) in index.endpoints


def methods_for(index: BenignIndex, host: str | None, port: int | None) -> set[str]:
    return set(index.methods.get((host, port), set()))


def binaries_for(index: BenignIndex, host: str | None, port: int | None) -> set[str]:
    return set(index.binaries.get((host, port), set()))


def trie_for(index: BenignIndex, host: str | None, port: int | None) -> PathTrie:
    return index.tries.get((host, port), PathTrie())


def semantic_fields_for(index: BenignIndex, host: str | None, port: int | None) -> set[str]:
    return set(index.semantic_fields.get((host, port), set()))


def queries_for(index: BenignIndex, host: str | None, port: int | None) -> set[str]:
    return set(index.queries.get((host, port), set()))


def sources_for(index: BenignIndex, host: str | None, port: int | None) -> list[str]:
    return list(index.sources.get((host, port), []))
