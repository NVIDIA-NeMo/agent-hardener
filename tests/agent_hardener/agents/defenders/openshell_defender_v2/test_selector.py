# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``analysis.selector.select`` picks the highest-priority feasible node and orders fallbacks."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.analysis.benign_index import (
    build_index,
    merge_counterexamples,
)
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.overlap import compute_overlap
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.selector import NODE_PRIORITY, select
from agent_hardener.agents.defenders.openshell_defender_v2.models import (
    BenignPrediction,
    DefenderContext,
    RequestTuple,
)
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml
from agent_hardener.models.contracts import DefenderInput

pytestmark = pytest.mark.unit

_POLICY = """\
version: 1
network_policies:
  github:
    name: github
    endpoints:
    - host: api.github.com
      port: 443
      protocol: http
      rules:
      - allow: {method: GET, path: /repos/*/*/issues}
"""


def _ctx(**overrides) -> DefenderContext:
    policy = load_policy_from_yaml(_POLICY)
    cut = overrides.pop(
        "cut",
        RequestTuple(host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"),
    )
    predictions = overrides.pop("benign_predictions", [])
    benign_index = build_index(predictions)
    return DefenderContext(
        defender_input=DefenderInput(attack_prompt="", agent_response="", attacked_tool="t"),
        policy=policy,
        cut=cut,
        benign_index=benign_index,
        overlap=compute_overlap(cut, benign_index),
        **overrides,
    )


def test_add_deny_rule_chosen_when_path_is_cleanly_separable_but_endpoint_is_touched() -> None:
    benign = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")
    predictions = [BenignPrediction(source_request="list issues", tuples=[benign])]
    selection = select(_ctx(benign_predictions=predictions))
    assert selection.chosen == "add_deny_rule"
    assert selection.infeasible  # remove_endpoint still recorded its reason


def test_remove_endpoint_chosen_when_host_port_entirely_untouched_by_benign() -> None:
    selection = select(_ctx())  # no benign predictions at all -> host:port entirely untouched
    assert selection.chosen == "remove_endpoint"
    # add_deny_rule is also feasible here (an empty trie makes every prefix "clean"), so it
    # shows up as a fallback rather than an infeasibility reason.
    assert [name for name, _ in selection.fallbacks] == ["add_deny_rule"]


def test_remove_endpoint_infeasible_when_endpoint_touched_by_a_bare_tuple() -> None:
    """A bare host/port tuple (no method/path/binary/semantic field) must still register the

    endpoint as touched, so ``remove_endpoint`` doesn't treat it as "entirely absent from
    predicted benign traffic".
    """
    policy = load_policy_from_yaml(_POLICY)
    cut = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    benign_index = build_index([])
    merge_counterexamples(benign_index, [RequestTuple(host="api.github.com", port=443, protocol="http")])
    ctx = DefenderContext(
        defender_input=DefenderInput(attack_prompt="", agent_response="", attacked_tool="t"),
        policy=policy,
        cut=cut,
        benign_index=benign_index,
        overlap=compute_overlap(cut, benign_index),
    )

    selection = select(ctx)

    assert selection.chosen != "remove_endpoint"
    assert "remove_endpoint" in selection.infeasible


def test_both_nodes_infeasible_when_endpoint_is_in_protected_endpoints() -> None:
    """``protected_endpoints`` is a stronger guarantee than the benign index: even an attack whose

    cut has a cleanly-separable path (which would otherwise make ``add_deny_rule`` feasible, per
    ``test_remove_endpoint_infeasible_when_endpoint_touched_by_a_bare_tuple`` above) must not get
    narrowed either — the endpoint may not be touched by any node, full stop.
    """
    ctx = _ctx(protected_endpoints={("api.github.com", 443)})

    selection = select(ctx)

    assert selection.chosen is None
    assert "remove_endpoint" in selection.infeasible
    assert "add_deny_rule" in selection.infeasible


def test_both_nodes_infeasible_when_protected_endpoint_matches_via_wildcard_port() -> None:
    """The exact reported regression: an attack whose extracted cut has ``port=None`` (not

    explicit in the attack prompt) still resolves to the concrete protected endpoint via
    ``_locate``'s ``None``-port wildcard — the protection check must use the same wildcard
    semantics, or it silently misses this case while the node still fires.
    """
    cut = RequestTuple(
        host="api.github.com", port=None, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    ctx = _ctx(cut=cut, protected_endpoints={("api.github.com", 443)})

    selection = select(ctx)

    assert selection.chosen is None
    assert "remove_endpoint" in selection.infeasible
    assert "add_deny_rule" in selection.infeasible


def test_excluded_node_falls_through_to_next_priority() -> None:
    selection = select(_ctx(), excluded={"remove_endpoint"})
    assert selection.chosen != "remove_endpoint"
    assert "remove_endpoint" in selection.infeasible


def test_no_feasible_node_returns_empty_selection() -> None:
    ctx = _ctx(cut=RequestTuple())  # no host at all -> nothing can locate a target
    selection = select(ctx)
    assert selection.chosen is None
    assert selection.witness is None
    assert set(selection.infeasible) == set(NODE_PRIORITY)


def test_fallback_ordering_matches_node_priority() -> None:
    selection = select(_ctx())
    fallback_names = [name for name, _ in selection.fallbacks]
    priority_order = [n for n in NODE_PRIORITY if n == selection.chosen or n in fallback_names]
    assert priority_order == [selection.chosen, *fallback_names]
