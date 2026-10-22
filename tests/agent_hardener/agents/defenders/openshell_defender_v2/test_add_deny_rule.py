# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``nodes.add_deny_rule.synthesize`` formats a witness into a deny-rule candidate that only contracts access."""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2.models import (
    BenignPrediction,
    DefenderContext,
    RequestTuple,
    Witness,
)
from agent_hardener.agents.defenders.openshell_defender_v2.nodes.add_deny_rule import (
    NODE,
    AddDenyRuleWitness,
    synthesize,
)
from agent_hardener.agents.defenders.openshell_defender_v2.policy.lint import assert_contraction
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml
from agent_hardener.agents.defenders.openshell_defender_v2.policy.patch import AddDenyRule, PolicyDelta, apply_delta
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
      protocol: rest
      enforcement: enforce
      allow_encoded_slash: true
      rules:
      - allow: {method: GET, path: /repos/*/*/issues}
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

_ENDPOINTS = {"github": 0, "gql": 0, "mcp": 0, "rpc": 0}
_HOSTS = {"github": "api.github.com", "gql": "gql.example.com", "mcp": "mcp.example.com", "rpc": "rpc.example.com"}


def _ctx(entry_key: str = "github", benign: list[RequestTuple] | None = None, requests: int = 0) -> DefenderContext:
    host = _HOSTS[entry_key]
    tuples = benign or []
    predictions = [
        BenignPrediction(source_request=f"req-{i}", tuples=[RequestTuple(host=host, port=443)]) for i in range(requests)
    ]
    return DefenderContext(
        defender_input=DefenderInput(attack_prompt="", agent_response="", attacked_tool="t"),
        policy=load_policy_from_yaml(_POLICY),
        benign_tuples=tuples,
        benign_predictions=predictions,
    )


def _witness(entry_key: str = "github", **fields) -> AddDenyRuleWitness:
    return AddDenyRuleWitness(entry_key=entry_key, endpoint_index=_ENDPOINTS[entry_key], **fields)


def _rule(candidate):
    (op,) = PolicyDelta.model_validate(candidate.delta).ops
    assert isinstance(op, AddDenyRule)
    return op.rule


def test_rest_witness_denies_every_method_on_the_glob() -> None:
    candidate = synthesize(_ctx(requests=3), _witness(glob="/admin/**", attack_methods={"DELETE"}))

    rule = _rule(candidate)
    assert (rule.method, rule.path) == ("*", "/admin/**")  # pinning to DELETE would be bypassable via PUT
    assert candidate.node == "add_deny_rule"
    assert candidate.resource_type == "l7_deny_rule"
    assert candidate.confidence == "high"
    assert "provably touches no predicted benign path" in candidate.rationale


def test_rest_candidate_is_a_pure_contraction_that_keeps_the_allow_set() -> None:
    ctx = _ctx()
    candidate = synthesize(ctx, _witness(glob="/admin/**"))

    after = apply_delta(ctx.policy, PolicyDelta.model_validate(candidate.delta))

    assert_contraction(ctx.policy, after)  # raises if the deny rule widened access
    before_ep = ctx.policy.network_policies["github"].endpoints[0]
    after_ep = after.network_policies["github"].endpoints[0]
    assert after_ep.rules == before_ep.rules
    assert after_ep.access == before_ep.access
    assert [r.path for r in after_ep.deny_rules] == ["/admin/**"]
    assert before_ep.deny_rules is None  # the input policy is never mutated


def test_rest_confidence_is_low_when_the_benign_corpus_is_thin() -> None:
    assert synthesize(_ctx(requests=2), _witness(glob="/admin/**")).confidence == "low"
    assert synthesize(_ctx(requests=3), _witness(glob="/admin/**")).confidence == "high"


def test_encoded_slash_endpoint_is_flagged_but_left_untouched() -> None:
    ctx = _ctx()
    candidate = synthesize(ctx, _witness(glob="/admin/**"))

    assert "allow_encoded_slash left untouched" in candidate.rationale
    after = apply_delta(ctx.policy, PolicyDelta.model_validate(candidate.delta))
    assert after.network_policies["github"].endpoints[0].allow_encoded_slash is True


def test_graphql_witness_denies_the_field_under_its_operation_type() -> None:
    witness = _witness(
        "gql", protocol="graphql", field_kind="graphql_field", field_value="deleteRepo", operation_type="mutation"
    )

    candidate = synthesize(_ctx("gql", requests=3), witness)

    rule = _rule(candidate)
    assert (rule.operation_type, rule.fields) == ("mutation", ["deleteRepo"])
    assert rule.path is None
    assert candidate.confidence == "high"


def test_mcp_witness_denies_the_tool_via_tools_call() -> None:
    witness = _witness("mcp", protocol="mcp", field_kind="mcp_tool", field_value="delete_file")

    rule = _rule(synthesize(_ctx("mcp", requests=3), witness))

    assert (rule.method, rule.tool) == ("tools/call", "delete_file")


def test_jsonrpc_witness_uses_an_exact_method_string() -> None:
    witness = _witness("rpc", protocol="json-rpc", field_kind="jsonrpc_method", field_value="admin.drop")

    rule = _rule(synthesize(_ctx("rpc", requests=3), witness))

    assert (rule.method, rule.tool, rule.path) == ("admin.drop", None, None)


@pytest.mark.parametrize(
    ("entry_key", "protocol", "kind"),
    [("mcp", "mcp", "mcp_tool"), ("rpc", "json-rpc", "jsonrpc_method")],
)
def test_batch_collateral_forces_low_confidence_even_with_a_rich_corpus(
    entry_key: str, protocol: str, kind: str
) -> None:
    # Batches are denied wholesale, so a benign request that shares the tool/method is collateral damage.
    benign = [RequestTuple(host=_HOSTS[entry_key], port=443, mcp_tool="shared_tool")]
    witness = _witness(entry_key, protocol=protocol, field_kind=kind, field_value="shared_tool")

    candidate = synthesize(_ctx(entry_key, benign=benign, requests=10), witness)

    assert candidate.confidence == "low"
    assert "may also deny benign batches" in candidate.rationale


def test_graphql_never_gets_a_batch_collateral_warning() -> None:
    benign = [RequestTuple(host="gql.example.com", port=443, mcp_tool="deleteRepo")]
    witness = _witness("gql", protocol="graphql", field_kind="graphql_field", field_value="deleteRepo")

    candidate = synthesize(_ctx("gql", benign=benign, requests=10), witness)

    assert candidate.confidence == "high"
    assert "batches" not in candidate.rationale


def test_unrelated_benign_tuples_do_not_trigger_the_collateral_warning() -> None:
    benign = [RequestTuple(host="mcp.example.com", port=443, mcp_tool="read_file")]
    witness = _witness("mcp", protocol="mcp", field_kind="mcp_tool", field_value="delete_file")

    assert synthesize(_ctx("mcp", benign=benign, requests=10), witness).confidence == "high"


def test_witness_pointing_at_a_missing_target_raises_instead_of_patching_blindly() -> None:
    with pytest.raises(KeyError):
        synthesize(_ctx(), AddDenyRuleWitness(entry_key="nope", endpoint_index=0, glob="/x/**"))
    with pytest.raises(IndexError):
        synthesize(_ctx(), AddDenyRuleWitness(entry_key="github", endpoint_index=5, glob="/x/**"))


def test_foreign_witness_type_is_rejected() -> None:
    with pytest.raises(AssertionError):
        synthesize(_ctx(), Witness(node="remove_endpoint"))


def test_node_only_allows_the_add_deny_rule_op() -> None:
    assert NODE.allowed_ops == frozenset({AddDenyRule})
