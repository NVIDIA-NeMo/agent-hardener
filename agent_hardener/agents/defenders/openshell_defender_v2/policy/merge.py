# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Merge N independently-computed candidate policies into one.

Concurrent defenders each compute a candidate against the same pristine policy snapshot, with no
visibility into each other's edits. Without a merge step, writing every candidate to the one
shared active-state file degenerates to "whichever defender finishes last wins" — silently
discarding every other candidate's fix. See ``nodes/{remove_endpoint,add_deny_rule}.py`` for the
only two edit kinds a node can produce today (both strict contractions of the original policy),
which is what makes the merge below fully mechanical.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..config import DefenderConfig, load_config
from ..llm.client import complete_structured
from .lint import assert_contraction

if TYPE_CHECKING:
    from .schema import DenyRule, Endpoint, Policy

EndpointKey = tuple[str, str, int | None]  # (entry_key, host, port)


def _endpoint_map(policy: Policy) -> dict[EndpointKey, Endpoint]:
    return {
        (entry_key, ep.host, ep.port): ep
        for entry_key, entry in policy.network_policies.items()
        for ep in entry.endpoints
    }


def _out_of_scope_drift(original: Policy, candidate: Policy) -> bool:
    """True if ``candidate`` changed anything outside ``network_policies`` — no current node does

    this; treated as an anomaly and excluded from the merge rather than silently trusted.
    """
    exclude = {"network_policies"}
    return original.model_dump(mode="json", exclude=exclude) != candidate.model_dump(mode="json", exclude=exclude)


def _added_deny_rules(original_ep: Endpoint | None, candidate_ep: Endpoint) -> list[DenyRule]:
    original_rules = (original_ep.deny_rules or []) if original_ep else []
    return [r for r in (candidate_ep.deny_rules or []) if r not in original_rules]


def _is_unexpected_diff(original_ep: Endpoint, candidate_ep: Endpoint) -> bool:
    """True if ``candidate_ep`` differs from ``original_ep`` in anything besides appended

    ``deny_rules`` — a shape no current node produces; only relevant as a forward-compat guard.
    """
    return original_ep.model_copy(update={"deny_rules": candidate_ep.deny_rules}) != candidate_ep


class MergeConflictOutput(BaseModel):
    """Which candidate endpoint config to prefer when candidates disagree outside the

    remove/add-deny-rule vocabulary this merge otherwise resolves mechanically.
    """

    candidate_index: int  # index into the conflicting candidate list; out-of-range = keep original


MERGE_CONFLICT_PROMPT = """\
Multiple independent security patches narrowed the same network policy endpoint in ways that
don't reduce to "removed" or "added a deny rule" — an unusual shape. The original endpoint:
{original}

Candidate versions of this endpoint:
{candidates}

Return the index of the candidate whose version is the SAFEST (most restrictive / narrowest)
choice to adopt. If you are not confident any of them is safe, return -1 to keep the original
endpoint unchanged rather than guess.
"""


def _resolve_conflict(original_ep: Endpoint, candidate_eps: list[Endpoint], config: DefenderConfig) -> Endpoint | None:
    """Only ``deny_rules`` is ever adopted from the LLM's chosen candidate — every other field is

    forced back to ``original_ep``, and the caller re-runs ``assert_contraction`` on the whole
    merged policy regardless, so a bad answer can under-narrow but never widen access.
    """
    prompt = MERGE_CONFLICT_PROMPT.format(
        original=original_ep.model_dump(mode="json", exclude_none=True),
        candidates="\n".join(
            f"{i}: {ep.model_dump(mode='json', exclude_none=True)}" for i, ep in enumerate(candidate_eps)
        ),
    )
    output = complete_structured(prompt, MergeConflictOutput, config)
    if not (0 <= output.candidate_index < len(candidate_eps)):
        return None
    return original_ep.model_copy(update={"deny_rules": candidate_eps[output.candidate_index].deny_rules})


def _collect_votes(
    original_map: dict[EndpointKey, Endpoint], usable: list[Policy]
) -> tuple[set[EndpointKey], dict[EndpointKey, list[DenyRule]], dict[EndpointKey, list[Endpoint]]]:
    """One pass over every candidate's opinion of every original endpoint: removed, narrowed

    (deny rules added), or an unexpected diff outside that vocabulary.
    """
    removed: set[EndpointKey] = set()
    added_by_key: dict[EndpointKey, list[DenyRule]] = defaultdict(list)
    conflicts: dict[EndpointKey, list[Endpoint]] = defaultdict(list)

    for key, original_ep in original_map.items():
        for candidate in usable:
            candidate_ep = _endpoint_map(candidate).get(key)
            if candidate_ep is None:
                removed.add(key)
            elif _is_unexpected_diff(original_ep, candidate_ep):
                conflicts[key].append(candidate_ep)
            else:
                added_by_key[key].extend(_added_deny_rules(original_ep, candidate_ep))

    return removed, added_by_key, conflicts


def merge_candidates(original: Policy, candidates: list[Policy], config: DefenderConfig | None = None) -> Policy:
    """Merge N independently-computed candidate policies into one: removal beats narrowing,

    narrowing beats no-op, no-op only when every candidate left the endpoint untouched. Raises
    ``LintViolationError`` (via ``assert_contraction``) if the merged result is somehow not a
    strict contraction of ``original`` — should never happen by construction; a caller-visible
    safety net, not an expected path.
    """
    usable = [c for c in candidates if not _out_of_scope_drift(original, c)]

    original_map = _endpoint_map(original)
    merged = original.model_copy(deep=True)
    merged_map = _endpoint_map(merged)

    removed, added_by_key, conflicts = _collect_votes(original_map, usable)

    for key, candidate_eps in conflicts.items():
        if key in removed:
            continue
        resolved = _resolve_conflict(original_map[key], candidate_eps, config or load_config())
        if resolved is not None:
            merged_map[key].deny_rules = resolved.deny_rules

    for entry_key, host, port in removed:
        entry = merged.network_policies.get(entry_key)
        if entry is None:
            continue
        entry.endpoints = [ep for ep in entry.endpoints if (ep.host, ep.port) != (host, port)]
        if not entry.endpoints:
            del merged.network_policies[entry_key]

    for key, new_rules in added_by_key.items():
        if not new_rules or key in removed or key not in merged_map:
            continue
        existing = merged_map[key].deny_rules or []
        existing.extend(r for r in new_rules if r not in existing)
        merged_map[key].deny_rules = existing

    assert_contraction(original, merged)
    return merged
