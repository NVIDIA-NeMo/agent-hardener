# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end ``run(DefenderInput) -> DefenderOutput`` with the sole LLM boundary stubbed.

Stubbing ``llm.client.complete_structured``/``complete_batch`` (imported directly into each
calling module, so patched at their use sites) exercises the entire deterministic pipeline for
real: extraction plumbing, overlap computation, the feasibility/selector chain, node synthesis,
and the lint gate.
"""

from __future__ import annotations

import pytest

from agent_hardener.agents.defenders.openshell_defender_v2 import openshell_defender_agent as agent
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.prompts import (
    AttackTuplesOutput,
    BenignTuplesOutput,
)
from agent_hardener.agents.defenders.openshell_defender_v2.models import HarmCertificate, RequestTuple
from agent_hardener.models.contracts import DefenderInput, ValidationFeedback
from agent_hardener.models.infra import BackendEndpoint, BackendServiceSpec, RelayVictimSpec

pytestmark = pytest.mark.unit

_POLICY_YAML = """\
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


def _stub_llm(monkeypatch, *, attack_tuples, harm_cert, benign_tuples):
    def fake_complete_structured(prompt, schema, config):
        if schema is AttackTuplesOutput:
            return AttackTuplesOutput(tuples=attack_tuples)
        if schema is HarmCertificate:
            return harm_cert
        if schema is BenignTuplesOutput:
            return BenignTuplesOutput(tuples=benign_tuples)
        raise AssertionError(f"unexpected schema {schema}")

    def fake_complete_batch(prompts, schema, config):
        return [fake_complete_structured(p, schema, config) for p in prompts]

    monkeypatch.setattr(
        "agent_hardener.agents.defenders.openshell_defender_v2.extraction.attack.complete_structured",
        fake_complete_structured,
    )
    monkeypatch.setattr(
        "agent_hardener.agents.defenders.openshell_defender_v2.extraction.benign.complete_batch", fake_complete_batch
    )
    monkeypatch.setattr(
        "agent_hardener.agents.defenders.openshell_defender_v2.openshell_defender_agent.inference_api_key",
        lambda: "fake-key",
    )


def _run(**kwargs) -> DefenderInput:
    return DefenderInput(
        attack_prompt=kwargs.pop("attack_prompt", "delete the issue"),
        agent_response=kwargs.pop("agent_response", "DELETE /repos/foo/bar/issues/1"),
        attacked_tool=kwargs.pop("attacked_tool", "github_tool"),
        benign_requests=kwargs.pop("benign_requests", []),
        context={"current_policy": _POLICY_YAML, **kwargs.pop("context", {})},
        **kwargs,
    )


def test_gate_abstains_out_of_scope_when_attack_has_no_network_signature(monkeypatch) -> None:
    _stub_llm(
        monkeypatch,
        attack_tuples=[RequestTuple(binary="/bin/cat", path="/etc/passwd")],
        harm_cert=HarmCertificate(harm_class="file_read", channel="file_read", cut_index=0),
        benign_tuples=[],
    )
    output = agent.run(_run())
    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("out_of_scope")


def test_no_feasible_node_abstains(monkeypatch) -> None:
    # Attack and benign hit the exact same tuple on every axis -> nothing can separate them.
    same = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")
    _stub_llm(
        monkeypatch,
        attack_tuples=[same],
        harm_cert=HarmCertificate(harm_class="data_exfiltration", channel="http_response_body", cut_index=0),
        benign_tuples=[same],
    )
    output = agent.run(_run(benign_requests=["list issues"]))
    assert output.ok is False
    assert output.new_policy_yaml is None


def test_produces_a_lint_clean_patch_when_a_clean_carveout_exists(monkeypatch) -> None:
    attack = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    benign = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="unauthorized_write", channel="http_response_body", cut_index=0),
        benign_tuples=[benign],
    )
    output = agent.run(_run(benign_requests=["list open issues on foo/bar"]))
    assert output.ok is True
    assert output.resource_type == "l7_deny_rule"
    assert output.new_policy_yaml is not None
    assert "deny_rules" in output.new_policy_yaml


def _integration_backend_relay_victim_spec() -> RelayVictimSpec:
    return RelayVictimSpec(
        project_dir="/tmp/project",
        dockerfile="Dockerfile",
        victim_binaries=["/app/**"],
        backends=[
            BackendServiceSpec(
                name="integration_backend",
                allowlist=[BackendEndpoint(host="host.docker.internal", port=8089, protocol="rest")],
            )
        ],
    )


def test_protected_backend_endpoint_is_never_touched(monkeypatch) -> None:
    """An attack whose cut targets a relay_victim.backends[].allowlist endpoint must abstain

    outright — no remove_endpoint, no add_deny_rule — even though the attack has a perfectly
    valid, narrowly-separable network signature that would otherwise let add_deny_rule fire.
    """
    attack = RequestTuple(host="host.docker.internal", port=8089, protocol="rest", method="GET", path="/exfil")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="ssrf", channel="http_response_body", cut_index=0),
        benign_tuples=[],
    )
    output = agent.run(_run(relay_victim_spec=_integration_backend_relay_victim_spec()))
    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("protected_endpoint")


def test_protected_backend_endpoint_is_never_touched_when_attack_port_is_unspecified(monkeypatch) -> None:
    """The exact reported regression: an attack prompt that references the backend without

    literally stating its port produces ``cut.port=None``. ``_locate`` still resolves this to the
    concrete protected endpoint via its own ``None``-port wildcard, so the protection check must
    do the same rather than miss it on an exact-tuple comparison.
    """
    attack = RequestTuple(host="host.docker.internal", port=None, protocol="rest", method="GET", path="/exfil")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="ssrf", channel="http_response_body", cut_index=0),
        benign_tuples=[],
    )
    output = agent.run(_run(relay_victim_spec=_integration_backend_relay_victim_spec()))
    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("protected_endpoint")


def test_agent_env_endpoint_is_protected_even_without_a_matching_backend_entry(monkeypatch) -> None:
    """Every environment tuple (agent_env, not just backends[].allowlist/the LLM endpoint) is now

    part of the same absolute block-list — an agent_env URL is protected on its own, without
    needing a matching ``backends[].allowlist`` entry.
    """
    other_policy = _POLICY_YAML + (
        "  other:\n"
        "    name: other\n"
        "    endpoints:\n"
        "    - host: other.internal\n"
        "      port: 9000\n"
        "      protocol: http\n"
        "      access: full\n"
    )
    attack = RequestTuple(host="other.internal", port=9000, protocol="http")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="ssrf", channel="http_response_body", cut_index=0),
        benign_tuples=[],
    )
    relay_victim_spec = RelayVictimSpec(
        project_dir="/tmp/project",
        dockerfile="Dockerfile",
        victim_binaries=["/app/**"],
        agent_env={"OTHER_URL": "http://other.internal:9000"},
    )
    output = agent.run(_run(relay_victim_spec=relay_victim_spec, context={"current_policy": other_policy}))
    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("protected_endpoint")


def test_no_environment_endpoints_when_no_relay_victim_spec_attack_proceeds_normally(monkeypatch) -> None:
    """Sanity check: without a ``relay_victim_spec`` at all, ``protected_endpoints`` is empty and

    an otherwise-untouched host:port is still removable — the new machinery doesn't change
    behavior for defenders/targets that don't carry environment config.
    """
    attack = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="unauthorized_write", channel="http_response_body", cut_index=0),
        benign_tuples=[],
    )
    output = agent.run(_run())
    assert output.ok is True
    assert output.resource_type == "endpoint_removal"


def test_feedback_excludes_previously_failed_node(monkeypatch) -> None:
    attack = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    benign = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="unauthorized_write", channel="http_response_body", cut_index=0),
        benign_tuples=[benign],
    )
    # This round's starting policy already carries the previous (failed) add_deny_rule patch;
    # feedback.previous_policy_yaml is what it looked like *before* that patch was applied, so
    # the diff between them identifies add_deny_rule as the node to exclude this round. With only
    # two nodes left, excluding add_deny_rule leaves remove_endpoint as the sole fallback, which
    # isn't feasible here (benign traffic still touches this host:port) -> the round abstains
    # rather than silently re-emitting the same deny rule.
    policy_with_deny = _POLICY_YAML + "      deny_rules:\n      - {method: '*', path: /repos/foo/bar/issues/1/**}\n"
    feedback = ValidationFeedback(false_negatives=["attack still got through"], previous_policy_yaml=_POLICY_YAML)
    output = agent.run(
        _run(
            benign_requests=["list open issues on foo/bar"],
            feedback=feedback,
            context={"current_policy": policy_with_deny},
        )
    )
    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("no_feasible_node")
