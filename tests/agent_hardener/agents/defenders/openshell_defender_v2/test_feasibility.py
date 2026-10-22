# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``analysis.feasibility`` predicates: when each node may fire, and the witness it carries."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.analysis.benign_index import build_index
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.feasibility import (
    FEASIBILITY_CHECKS,
    check_add_deny_rule,
    check_remove_endpoint,
    infeasibility_reason,
)
from agent_hardener.agents.defenders.openshell_defender_v2.analysis.overlap import compute_overlap
from agent_hardener.agents.defenders.openshell_defender_v2.models import (
    BenignPrediction,
    DefenderContext,
    OverlapVector,
    RequestTuple,
)
from agent_hardener.agents.defenders.openshell_defender_v2.policy.globs import glob_matches
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml
from agent_hardener.models.contracts import DefenderInput

pytestmark = pytest.mark.unit

_POLICY = """\
version: 1
network_policies:
  github:
    name: github
    binaries:
    - path: /app/bin/agent
    endpoints:
    - host: api.github.com
      port: 443
      protocol: rest
      enforcement: enforce
      rules:
      - allow: {method: GET, path: /repos/*/*/issues}
  audit:
    name: audit
    endpoints:
    - {host: audit.example.com, port: 443, protocol: rest, access: full}
  passthrough:
    name: passthrough
    endpoints:
    - {host: raw.example.com, port: 443, protocol: tcp, tls: skip, enforcement: enforce}
  bare:
    name: bare
    endpoints:
    - {host: bare.example.com, port: 443, enforcement: enforce}
  gql:
    name: gql
    endpoints:
    - {host: gql.example.com, port: 443, protocol: graphql, enforcement: enforce, access: full}
  mcp:
    name: mcp
    endpoints:
    - {host: mcp.example.com, port: 443, protocol: mcp, enforcement: enforce, access: full}
  rpc:
    name: rpc
    endpoints:
    - {host: rpc.example.com, port: 443, protocol: json-rpc, enforcement: enforce, access: full}
"""

_ATTACK = RequestTuple(
    host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
)
_BENIGN_LIST = RequestTuple(
    host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues"
)


def _ctx(cut: RequestTuple | None = _ATTACK, benign: list[RequestTuple] | None = None, **overrides) -> DefenderContext:
    predictions = [BenignPrediction(source_request="b", tuples=benign)] if benign else []
    index = build_index(predictions)
    kwargs = {
        "benign_index": index,
        "overlap": compute_overlap(cut, index) if cut else None,
        **overrides,
    }
    return DefenderContext(
        defender_input=DefenderInput(attack_prompt="", agent_response="", attacked_tool="t"),
        policy=load_policy_from_yaml(_POLICY),
        cut=cut,
        **kwargs,
    )


# --- add_deny_rule: gating ---


@pytest.mark.parametrize(
    "cut",
    [
        None,
        RequestTuple(binary="/bin/cat", path="/etc/passwd"),  # no host: no network signature
        RequestTuple(host="unknown.example.com", port=443, path="/x"),
        RequestTuple(host="api.github.com", port=443, binary="/other/bin", path="/x"),  # binary not in entry
    ],
    ids=["no-cut", "no-host", "host-not-in-policy", "binary-mismatch"],
)
def test_add_deny_rule_needs_a_cut_that_resolves_to_a_policy_endpoint(cut) -> None:
    assert check_add_deny_rule(_ctx(cut)) is None


def test_add_deny_rule_never_fires_on_a_protected_endpoint() -> None:
    assert check_add_deny_rule(_ctx(protected_endpoints={("api.github.com", 443)})) is None
    assert check_add_deny_rule(_ctx(protected_endpoints={("api.github.com", None)})) is None
    assert check_add_deny_rule(_ctx(protected_endpoints={("other.example.com", 443)})) is not None


def test_add_deny_rule_refuses_audit_endpoints_where_a_deny_would_only_log() -> None:
    cut = RequestTuple(host="audit.example.com", port=443, protocol="http", path="/danger")

    assert check_add_deny_rule(_ctx(cut)) is None


# --- add_deny_rule: REST branch ---


def test_rest_witness_glob_covers_the_attack_and_no_benign_path() -> None:
    witness = check_add_deny_rule(_ctx(benign=[_BENIGN_LIST]))

    assert witness is not None
    assert (witness.entry_key, witness.endpoint_index) == ("github", 0)
    assert witness.attack_methods == {"DELETE"}
    assert not glob_matches(witness.glob, _BENIGN_LIST.path)


@pytest.mark.xfail(
    strict=True,
    reason="prefix_to_glob drops the leading '/', so the deny glob never matches real request paths "
    "(policy/globs.py:43; the trie strips it in analysis/trie.py:_segments)",
)
def test_rest_witness_glob_matches_the_attack_path() -> None:
    cut = RequestTuple(host="api.github.com", port=443, protocol="http", method="DELETE", path="/admin/delete")

    witness = check_add_deny_rule(_ctx(cut, benign=[_BENIGN_LIST]))

    assert glob_matches(witness.glob, "/admin/delete")


@pytest.mark.xfail(
    strict=True,
    reason="a full-path prefix becomes '<path>/**', which needs a trailing '/' and so misses the attack path itself "
    "(policy/globs.py:43-46)",
)
def test_rest_witness_glob_matches_the_exact_attack_path_when_only_the_full_path_is_clean() -> None:
    witness = check_add_deny_rule(_ctx(benign=[_BENIGN_LIST]))

    assert glob_matches(witness.glob, "/" + "/".join(["repos", "foo", "bar", "issues", "1"]))


def test_rest_witness_without_a_cut_method_has_no_attack_methods() -> None:
    cut = RequestTuple(host="api.github.com", port=443, protocol="http", path="/admin/x")

    assert check_add_deny_rule(_ctx(cut)).attack_methods == set()


def test_rest_abstains_when_attack_path_equals_a_benign_path() -> None:
    cut = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")

    assert check_add_deny_rule(_ctx(cut, benign=[_BENIGN_LIST])) is None


def test_rest_abstains_when_a_benign_path_lives_under_the_attack_path() -> None:
    cut = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo")

    assert check_add_deny_rule(_ctx(cut, benign=[_BENIGN_LIST])) is None


@pytest.mark.parametrize(
    "cut",
    [
        RequestTuple(host="api.github.com", port=443, method="DELETE"),  # no path
        RequestTuple(host="api.github.com", port=443, method="DELETE", path=""),
        RequestTuple(host="bare.example.com", port=443, path="/x"),  # endpoint has no L7 protocol
        RequestTuple(host="raw.example.com", port=443, path="/x"),  # tls: skip passthrough
    ],
    ids=["no-path", "empty-path", "no-protocol", "tls-skip"],
)
def test_rest_needs_a_path_and_an_inspectable_endpoint(cut) -> None:
    assert check_add_deny_rule(_ctx(cut)) is None


def test_rest_needs_a_benign_index() -> None:
    assert check_add_deny_rule(_ctx(benign_index=None)) is None


# --- add_deny_rule: semantic branch ---


@pytest.mark.parametrize(
    ("host", "cut_fields", "kind", "value"),
    [
        ("gql.example.com", {"gql_field": "deleteRepo"}, "graphql_field", "deleteRepo"),
        ("mcp.example.com", {"mcp_tool": "delete_file"}, "mcp_tool", "delete_file"),
        ("rpc.example.com", {"mcp_tool": "admin.drop"}, "jsonrpc_method", "admin.drop"),
    ],
)
def test_semantic_witness_names_the_separating_field(host, cut_fields, kind, value) -> None:
    cut = RequestTuple(host=host, port=443, **cut_fields)

    witness = check_add_deny_rule(_ctx(cut))

    assert (witness.field_kind, witness.field_value) == (kind, value)
    assert witness.glob is None


def test_semantic_witness_is_blocked_when_benign_traffic_uses_the_same_field() -> None:
    cut = RequestTuple(host="mcp.example.com", port=443, mcp_tool="delete_file")
    benign = [RequestTuple(host="mcp.example.com", port=443, mcp_tool="delete_file")]

    assert check_add_deny_rule(_ctx(cut, benign=benign)) is None


def test_semantic_witness_needs_a_field_and_an_overlap_vector() -> None:
    cut = RequestTuple(host="mcp.example.com", port=443)
    assert check_add_deny_rule(_ctx(cut)) is None

    cut = RequestTuple(host="mcp.example.com", port=443, mcp_tool="delete_file")
    assert check_add_deny_rule(_ctx(cut, overlap=None)) is None
    assert check_add_deny_rule(_ctx(cut, overlap=OverlapVector(semantic_field_collision=True))) is None


# --- remove_endpoint ---


def test_remove_endpoint_fires_when_no_benign_traffic_touches_the_endpoint() -> None:
    witness = check_remove_endpoint(_ctx())

    assert (witness.entry_key, witness.endpoint_index, witness.host, witness.port) == (
        "github",
        0,
        "api.github.com",
        443,
    )


def test_remove_endpoint_ignores_unrelated_benign_traffic() -> None:
    other = RequestTuple(host="gql.example.com", port=443, protocol="http", method="GET", path="/x")

    assert check_remove_endpoint(_ctx(benign=[other])) is not None


@pytest.mark.parametrize(
    "benign",
    [
        RequestTuple(host="api.github.com", port=443, method="GET"),
        RequestTuple(host="api.github.com", port=443, path="/repos"),
        RequestTuple(host="api.github.com", port=443, binary="/app/bin/agent"),
        RequestTuple(host="api.github.com", port=443, mcp_tool="t"),
        RequestTuple(host="api.github.com", port=443),
    ],
    ids=["method", "path", "binary", "semantic", "bare"],
)
def test_remove_endpoint_blocked_by_any_kind_of_benign_contact(benign) -> None:
    assert check_remove_endpoint(_ctx(benign=[benign])) is None


def test_remove_endpoint_blocked_by_protection_or_missing_index_or_unlocatable_cut() -> None:
    assert check_remove_endpoint(_ctx(protected_endpoints={("api.github.com", 443)})) is None
    assert check_remove_endpoint(_ctx(benign_index=None)) is None
    assert check_remove_endpoint(_ctx(RequestTuple(host="unknown.example.com", port=443))) is None
    assert check_remove_endpoint(_ctx(None)) is None


# --- registry / reasons ---


def test_registry_and_reasons_cover_both_nodes() -> None:
    assert set(FEASIBILITY_CHECKS) == {"remove_endpoint", "add_deny_rule"}
    assert "enforce" in infeasibility_reason("add_deny_rule", _ctx())
    assert "benign traffic" in infeasibility_reason("remove_endpoint", _ctx())
    assert infeasibility_reason("mystery", _ctx()) == "not feasible"
