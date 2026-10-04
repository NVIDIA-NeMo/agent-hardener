# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interpret ``DefenderInput.feedback: ValidationFeedback`` into pipeline directives.

``ValidationFeedback`` (defined in ``agent_hardener.models.contracts`` — not modified here) only
carries ``false_negatives``/``false_positives``/``previous_policy_yaml``; it has no notion of
"which node produced the previous patch". So ``excluded_nodes`` is a best-effort structural diff
between the previous and current policy YAML rather than an exact lookup — good enough to avoid
retrying the exact same (deterministic) choice forever, not a certainty.

Critical ordering, per the spec: ``_build_context`` must call ``merge_counterexamples`` *before*
``compute_overlap``. A benign request the validator saw get wrongly blocked should change the
feasibility verdicts for this round, not merely cross one node off the excluded list — otherwise
round 2+ is just blind node-skipping instead of actually learning from the mistake.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .extraction.benign import predict_benign_tuples
from .policy.loader import current_policy_text, load_policy_from_yaml

if TYPE_CHECKING:
    from agent_hardener.models.contracts import DefenderInput, ValidationFeedback

    from .config import DefenderConfig
    from .models import RequestTuple
    from .policy.schema import Endpoint, Policy


@dataclass
class FeedbackDirective:
    excluded_nodes: set[str] = field(default_factory=set)
    counterexample_tuples: list[RequestTuple] = field(default_factory=list)
    hint: str = ""


def _diff_endpoint(endpoint: Endpoint, prev_endpoint: Endpoint | None) -> str | None:
    """Which node's characteristic edit shape turned ``prev_endpoint`` into ``endpoint``."""
    prev_deny_count = len(prev_endpoint.deny_rules or []) if prev_endpoint else 0
    if endpoint.deny_rules and len(endpoint.deny_rules) > prev_deny_count:
        return "add_deny_rule"
    return None


def _infer_previous_node(previous: Policy, current: Policy) -> str | None:
    """Best-effort: which node produced ``current`` from ``previous``.

    Checked in ``NODE_PRIORITY`` order so the most specific signal wins first.
    """
    for entry_key, entry in current.network_policies.items():
        prev_entry = previous.network_policies.get(entry_key)
        for i, endpoint in enumerate(entry.endpoints):
            prev_endpoint = prev_entry.endpoints[i] if prev_entry and i < len(prev_entry.endpoints) else None
            found = _diff_endpoint(endpoint, prev_endpoint)
            if found:
                return found
        if prev_entry and len(entry.endpoints) < len(prev_entry.endpoints):
            return "remove_endpoint"
    if len(current.network_policies) < len(previous.network_policies):
        return "remove_endpoint"
    return None


def excluded_nodes(feedback: ValidationFeedback | None, current_policy_yaml: str) -> set[str]:
    """Nodes to skip this round because their previous patch (deterministically) failed.

    Failed either by not blocking the attack (``false_negatives``) or by over-blocking benign
    traffic (``false_positives``). Either way, re-selecting the same node would just reproduce
    the same patch, so it's excluded and the selector falls through to the next rung.
    """
    if feedback is None or feedback.previous_policy_yaml is None:
        return set()
    if not feedback.false_negatives and not feedback.false_positives:
        return set()
    try:
        previous = load_policy_from_yaml(feedback.previous_policy_yaml)
        current = load_policy_from_yaml(current_policy_yaml)
    except Exception:
        return set()
    node = _infer_previous_node(previous, current)
    return {node} if node else set()


def counterexamples_from(
    feedback: ValidationFeedback | None,
    policy: Policy,
    config: DefenderConfig | None = None,
) -> list[RequestTuple]:
    """Structurally extract tuples for every ``false_positives`` string.

    A validator directly observed these get wrongly blocked, so they're strictly more
    authoritative than a fresh prediction and must be folded into the benign index before this
    round's overlap computation.
    """
    if feedback is None or not feedback.false_positives:
        return []
    predictions = predict_benign_tuples(feedback.false_positives, config)
    return [t for prediction in predictions for t in prediction.tuples]


def interpret(
    feedback: ValidationFeedback | None,
    defender_input: DefenderInput,
    policy: Policy,
    config: DefenderConfig | None = None,
) -> FeedbackDirective:
    """Build the one directive object ``_build_context`` needs from raw ``feedback``."""
    return FeedbackDirective(
        excluded_nodes=excluded_nodes(feedback, current_policy_text(defender_input)),
        counterexample_tuples=counterexamples_from(feedback, policy, config),
        hint="; ".join(feedback.false_negatives) if feedback and feedback.false_negatives else "",
    )
