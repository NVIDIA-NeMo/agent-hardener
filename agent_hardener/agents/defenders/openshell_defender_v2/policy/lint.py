# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``assert_contraction`` — the hard gate every candidate patch must pass before it ships.

Structural, not prompt-based: catches a node writing an ``allow`` while claiming to narrow,
independent of how convincing its rationale text sounds.
"""

from __future__ import annotations

from collections import defaultdict

from ..errors import LintViolationError
from .globs import glob_is_subset
from .schema import Policy, effective_methods, effective_paths

_EndpointKey = tuple[str, int | None]


def _hosts(policy: Policy) -> set[str]:
    return {ep.host for entry in policy.network_policies.values() for ep in entry.endpoints}


def _binaries(policy: Policy) -> set[str]:
    return {b.path for entry in policy.network_policies.values() if entry.binaries for b in entry.binaries}


def _method_glob_pairs(policy: Policy) -> dict[_EndpointKey, set[tuple[str, str]]]:
    pairs: dict[_EndpointKey, set[tuple[str, str]]] = defaultdict(set)
    for entry in policy.network_policies.values():
        for ep in entry.endpoints:
            key = (ep.host, ep.port)
            methods = effective_methods(ep)
            paths = effective_paths(ep) or ["**"]
            for method in methods:
                for path in paths:
                    pairs[key].add((method, path))
    return dict(pairs)


def _endpoint_methods(policy: Policy) -> dict[_EndpointKey, set[str]]:
    out: dict[_EndpointKey, set[str]] = defaultdict(set)
    for entry in policy.network_policies.values():
        for ep in entry.endpoints:
            out[(ep.host, ep.port)] |= effective_methods(ep)
    return dict(out)


def _endpoint_paths(policy: Policy) -> dict[_EndpointKey, set[str]]:
    out: dict[_EndpointKey, set[str]] = defaultdict(set)
    for entry in policy.network_policies.values():
        for ep in entry.endpoints:
            out[(ep.host, ep.port)] |= set(effective_paths(ep))
    return dict(out)


def _new_hosts(before: Policy, after: Policy) -> list[str]:
    """Hosts present in ``after`` that ``before`` never granted access to at all."""
    return sorted(_hosts(after) - _hosts(before))


def _new_binaries(before: Policy, after: Policy) -> list[str]:
    """Binaries present in ``after`` that ``before`` never allowed to run at all."""
    return sorted(_binaries(after) - _binaries(before))


def _expanded_methods(before: Policy, after: Policy) -> list[str]:
    """``(host, port)`` pairs where ``after`` allows a method ``before`` didn't, for a host:port

    pair that existed in ``before`` (brand-new hosts are reported via ``_new_hosts`` instead).
    """
    before_methods = _endpoint_methods(before)
    after_methods = _endpoint_methods(after)
    violations = []
    for key, methods in after_methods.items():
        if key not in before_methods:
            continue
        extra = methods - before_methods[key]
        if extra:
            violations.append(f"{key[0]}:{key[1]} gained methods {sorted(extra)}")
    return violations


def _widened_globs(before: Policy, after: Policy) -> list[str]:
    """``(host, port)`` pairs where ``after`` allows a path not covered by any glob ``before``

    allowed, for a host:port pair that existed in ``before``.
    """
    before_paths = _endpoint_paths(before)
    after_paths = _endpoint_paths(after)
    violations = []
    for key, paths in after_paths.items():
        if key not in before_paths:
            continue
        before_set = before_paths[key]
        for path in paths:
            if any(glob_is_subset(path, bp) for bp in before_set):
                continue
            violations.append(f"{key[0]}:{key[1]} gained path {path}")
    return violations


def _new_allow_clauses(before: Policy, after: Policy) -> list[str]:
    """Catch-all: any ``(method, path)`` allow pair reachable in ``after`` that wasn't reachable

    in ``before``, for an endpoint that already existed. Subsumes the two checks above but is
    kept alongside them for tests that want to assert on one axis at a time.
    """
    before_pairs = _method_glob_pairs(before)
    after_pairs = _method_glob_pairs(after)
    violations = []
    for key, pairs in after_pairs.items():
        if key not in before_pairs:
            continue
        before_set = before_pairs[key]
        for method, path in pairs:
            if (method, path) in before_set:
                continue
            if any(m == method and glob_is_subset(path, bp) for m, bp in before_set):
                continue
            violations.append(f"{key[0]}:{key[1]} {method} {path}")
    return violations


def assert_contraction(before: Policy, after: Policy) -> None:
    """Raise :class:`LintViolationError` unless ``after`` is a strict contraction of ``before``:

    no new host, no new binary, no expanded method set, no widened glob, no new allow clause.
    """
    violations: list[str] = []
    violations += [f"new host: {h}" for h in _new_hosts(before, after)]
    violations += [f"new binary: {b}" for b in _new_binaries(before, after)]
    violations += _expanded_methods(before, after)
    violations += _widened_globs(before, after)
    violations += _new_allow_clauses(before, after)
    if violations:
        raise LintViolationError("candidate policy widens access: " + "; ".join(violations))
