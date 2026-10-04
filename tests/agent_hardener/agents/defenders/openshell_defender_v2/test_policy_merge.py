# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``policy.merge.merge_candidates``: N independently-computed candidates -> one merged policy.

Regression coverage for the incident where concurrent defenders, each computing a candidate
against the same pristine policy snapshot, clobbered each other on write — only the last-processed
candidate survived, silently discarding the others even though every one reported ``ok=True``.
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.config import DefenderConfig
from agent_hardener.agents.defenders.openshell_defender_v2.errors import LintViolationError
from agent_hardener.agents.defenders.openshell_defender_v2.policy.merge import MergeConflictOutput, merge_candidates
from agent_hardener.agents.defenders.openshell_defender_v2.policy.schema import DenyRule, Policy

pytestmark = pytest.mark.unit

_MODULE = "agent_hardener.agents.defenders.openshell_defender_v2.policy.merge"


def _base_policy() -> Policy:
    return Policy.model_validate(
        {
            "version": 1,
            "network_policies": {
                "beeceptor": {
                    "name": "beeceptor",
                    "endpoints": [
                        {"host": "abc123.free.beeceptor.com", "port": 443, "protocol": "http", "access": "full"},
                    ],
                },
                "integration_backend": {
                    "name": "integration_backend",
                    "endpoints": [
                        {"host": "host.docker.internal", "port": 8089, "protocol": "http", "access": "full"},
                    ],
                },
            },
        }
    )


def _remove_entry(policy: Policy, entry_key: str) -> Policy:
    candidate = policy.model_copy(deep=True)
    del candidate.network_policies[entry_key]
    return candidate


def _add_deny_rule(policy: Policy, entry_key: str, endpoint_index: int, rule: DenyRule) -> Policy:
    candidate = policy.model_copy(deep=True)
    endpoint = candidate.network_policies[entry_key].endpoints[endpoint_index]
    endpoint.deny_rules = [*(endpoint.deny_rules or []), rule]
    return candidate


def test_incident_reproduced_all_removals_survive() -> None:
    """Three candidates independently remove beeceptor, one removes the backend instead.

    Regardless of processing order, both removals must survive the merge.
    """
    base = _base_policy()
    beeceptor_removed = _remove_entry(base, "beeceptor")
    backend_removed = _remove_entry(base, "integration_backend")
    candidates = [beeceptor_removed, beeceptor_removed, beeceptor_removed, backend_removed]

    merged = merge_candidates(base, candidates)

    assert "beeceptor" not in merged.network_policies
    assert "integration_backend" not in merged.network_policies


def test_removal_wins_over_narrowing() -> None:
    base = _base_policy()
    removed = _remove_entry(base, "beeceptor")
    narrowed = _add_deny_rule(base, "beeceptor", 0, DenyRule(method="POST", path="/exfil"))

    merged = merge_candidates(base, [removed, narrowed])

    assert "beeceptor" not in merged.network_policies


def test_addition_wins_over_noop() -> None:
    base = _base_policy()
    narrowed = _add_deny_rule(base, "beeceptor", 0, DenyRule(method="POST", path="/exfil"))
    untouched = base.model_copy(deep=True)

    merged = merge_candidates(base, [narrowed, untouched])

    endpoint = merged.network_policies["beeceptor"].endpoints[0]
    assert endpoint.deny_rules == [DenyRule(method="POST", path="/exfil")]


def test_untouched_in_all_candidates_stays_unchanged() -> None:
    base = _base_policy()
    candidates = [base.model_copy(deep=True), base.model_copy(deep=True)]

    merged = merge_candidates(base, candidates)

    assert merged == base


def test_union_of_different_additive_rules() -> None:
    base = _base_policy()
    exfil_denied = _add_deny_rule(base, "beeceptor", 0, DenyRule(method="POST", path="/exfil"))
    backup_denied = _add_deny_rule(base, "beeceptor", 0, DenyRule(method="GET", path="/backup"))

    merged = merge_candidates(base, [exfil_denied, backup_denied])

    endpoint = merged.network_policies["beeceptor"].endpoints[0]
    assert DenyRule(method="POST", path="/exfil") in endpoint.deny_rules
    assert DenyRule(method="GET", path="/backup") in endpoint.deny_rules
    assert len(endpoint.deny_rules) == 2


def test_new_endpoint_only_in_a_candidate_is_ignored_not_merged_in() -> None:
    """A candidate that somehow introduced a brand-new endpoint (no current node does this) is

    silently dropped from the merge rather than adopted — the merge only ever narrows endpoints
    that existed in the original.
    """
    base = _base_policy()
    candidate = base.model_copy(deep=True)
    candidate.network_policies["new_entry"] = candidate.network_policies["beeceptor"].model_copy(deep=True)
    candidate.network_policies["new_entry"].endpoints[0].host = "evil.example.com"

    merged = merge_candidates(base, [candidate])

    assert "new_entry" not in merged.network_policies


def test_unexpected_diff_falls_back_to_llm_and_adopts_only_deny_rules(monkeypatch) -> None:
    """An endpoint diff outside remove/add-deny-rule (here: a changed ``access``) is a shape no

    current node produces. The LLM picks which candidate to prefer, but only its ``deny_rules``
    are ever adopted — every other field is forced back to the original, so even if the LLM's
    chosen candidate also widened ``access``, the merged result does not.
    """
    base = _base_policy()
    widened = base.model_copy(deep=True)
    endpoint = widened.network_policies["beeceptor"].endpoints[0]
    endpoint.access = "read-only"  # narrower access value, but NOT via deny_rules -> "unexpected"
    endpoint.deny_rules = [DenyRule(method="POST", path="/exfil")]

    monkeypatch.setattr(f"{_MODULE}.complete_structured", lambda *_args: MergeConflictOutput(candidate_index=0))

    merged = merge_candidates(base, [widened], DefenderConfig())

    endpoint = merged.network_policies["beeceptor"].endpoints[0]
    assert endpoint.access == "full"  # forced back to the original, not adopted from the candidate
    assert endpoint.deny_rules == [DenyRule(method="POST", path="/exfil")]  # only this was adopted


def test_unexpected_diff_out_of_range_index_keeps_original(monkeypatch) -> None:
    base = _base_policy()
    widened = base.model_copy(deep=True)
    widened.network_policies["beeceptor"].endpoints[0].access = "read-only"

    monkeypatch.setattr(f"{_MODULE}.complete_structured", lambda *_args: MergeConflictOutput(candidate_index=-1))

    merged = merge_candidates(base, [widened], DefenderConfig())

    assert merged.network_policies["beeceptor"].endpoints[0] == base.network_policies["beeceptor"].endpoints[0]


def test_out_of_scope_drift_excludes_the_whole_candidate() -> None:
    """A candidate that changed something outside ``network_policies`` (no current node does

    this) is excluded from the merge entirely rather than partially trusted.
    """
    base = _base_policy()
    drifted = base.model_copy(deep=True, update={"network_middlewares": {"a": 1}})
    removed_elsewhere = _remove_entry(base, "beeceptor")

    merged = merge_candidates(base, [drifted, removed_elsewhere])

    # The drifted candidate is excluded; the legitimate removal from the other candidate still applies.
    assert "beeceptor" not in merged.network_policies


def test_merge_never_widens_even_when_asked_to() -> None:
    """Defense-in-depth: ``assert_contraction`` runs on every merge result.

    The merge algorithm as designed cannot produce a widened policy through its public surface
    (new endpoints are ignored, and the LLM-conflict path only ever adopts ``deny_rules``), so
    this exercises the guarantee indirectly: a no-op merge is trivially its own contraction.
    """
    base = _base_policy()
    merged = merge_candidates(base, [base.model_copy(deep=True)])
    assert merged == base


def test_contraction_is_reused_from_lint(monkeypatch) -> None:
    """If a future bug ever let a merge widen access, ``assert_contraction`` (already proven

    correct by ``test_lint.py``) is what would catch it — confirm it's actually invoked by
    monkeypatching it to always raise and checking the exception propagates.
    """
    base = _base_policy()

    def _always_violates(_before, _after):
        raise LintViolationError("forced for test")

    monkeypatch.setattr(f"{_MODULE}.assert_contraction", _always_violates)

    with pytest.raises(LintViolationError):
        merge_candidates(base, [base.model_copy(deep=True)])
