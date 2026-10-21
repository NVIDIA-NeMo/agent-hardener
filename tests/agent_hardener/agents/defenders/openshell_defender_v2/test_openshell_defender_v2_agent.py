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
from agent_hardener.agents.defenders.openshell_defender_v2.config import load_config
from agent_hardener.agents.defenders.openshell_defender_v2.errors import ExtractionError, LintViolationError
from agent_hardener.agents.defenders.openshell_defender_v2.extraction.prompts import (
    AttackTuplesOutput,
    BenignTuplesOutput,
)
from agent_hardener.agents.defenders.openshell_defender_v2.feedback import FeedbackDirective
from agent_hardener.agents.defenders.openshell_defender_v2.models import (
    AbstainReason,
    DefenderContext,
    HarmCertificate,
    RequestTuple,
    Selection,
    Witness,
)
from agent_hardener.agents.defenders.openshell_defender_v2.policy.loader import load_policy_from_yaml
from agent_hardener.models.contracts import DefenderInput, DefenderOutput, ValidationFeedback
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
      protocol: rest
      enforcement: enforce
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
        "      protocol: rest\n"
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


# --- run() error handling and selection-loop fallbacks ---


def _ctx_for_loop(policy_yaml: str = _POLICY_YAML, cut: RequestTuple | None = None) -> DefenderContext:
    return DefenderContext(
        defender_input=_run(context={"current_policy": policy_yaml}),
        policy=load_policy_from_yaml(policy_yaml),
        cut=cut,
    )


def _witness(node: str) -> Witness:
    return Witness(node=node)


def test_run_requires_an_inference_api_key(monkeypatch) -> None:
    monkeypatch.setattr(agent, "inference_api_key", lambda: "")

    with pytest.raises(ValueError, match="INFERENCE_API_KEY"):
        agent.run(_run())


def test_run_abstains_with_extraction_failed_when_the_llm_extraction_breaks(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise ExtractionError("model returned garbage")

    monkeypatch.setattr(agent, "inference_api_key", lambda: "fake-key")
    monkeypatch.setattr(agent, "_build_context", boom)

    output = agent.run(_run())

    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message == "extraction_failed: model returned garbage"


def test_run_turns_unexpected_context_errors_into_a_failed_output(monkeypatch) -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("policy backend down")

    monkeypatch.setattr(agent, "inference_api_key", lambda: "fake-key")
    monkeypatch.setattr(agent, "_build_context", boom)

    output = agent.run(_run())

    assert (output.ok, output.error_message, output.iteration_count) == (False, "policy backend down", 0)


def test_run_turns_selection_loop_crashes_into_a_failed_output_with_the_iteration(monkeypatch) -> None:
    ctx = _ctx_for_loop(cut=RequestTuple(host="api.github.com", port=443))
    ctx.iteration = 4

    def boom(_ctx):
        raise RuntimeError("selector exploded")

    monkeypatch.setattr(agent, "inference_api_key", lambda: "fake-key")
    monkeypatch.setattr(agent, "_build_context", lambda *_a, **_k: ctx)
    monkeypatch.setattr(agent, "_run_selection_loop", boom)

    output = agent.run(_run())

    assert (output.ok, output.error_message, output.iteration_count) == (False, "selector exploded", 4)


def test_validator_counterexamples_are_merged_before_overlap_so_they_block_the_deny_carveout(monkeypatch) -> None:
    attack = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    benign = RequestTuple(host="api.github.com", port=443, protocol="http", method="GET", path="/repos/foo/bar/issues")
    # The validator saw the attack's own path wrongly blocked for benign traffic.
    counterexample = RequestTuple(
        host="api.github.com", port=443, protocol="http", method="DELETE", path="/repos/foo/bar/issues/1"
    )
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="unauthorized_write", channel="http_response_body", cut_index=0),
        benign_tuples=[benign],
    )
    monkeypatch.setattr(agent, "interpret", lambda *_a, **_k: FeedbackDirective(counterexample_tuples=[counterexample]))
    defender_input = _run(benign_requests=["list issues"])

    ctx = agent._build_context(defender_input, load_config())

    assert counterexample in ctx.benign_tuples
    assert ctx.benign_index.tries[("api.github.com", 443)].shallowest_clean_prefix(attack.path) is None
    assert agent.run(defender_input).ok is False  # no carve-out remains that spares the counterexample


def test_iteration_counts_every_piece_of_validator_feedback(monkeypatch) -> None:
    attack = RequestTuple(host="api.github.com", port=443, protocol="http", method="DELETE", path="/x")
    _stub_llm(
        monkeypatch,
        attack_tuples=[attack],
        harm_cert=HarmCertificate(harm_class="unauthorized_write", channel="http_response_body", cut_index=0),
        benign_tuples=[],
    )
    feedback = ValidationFeedback(false_negatives=["a", "b"], false_positives=["c"])

    ctx = agent._build_context(_run(feedback=feedback), load_config())

    assert ctx.iteration == 3


def test_selection_loop_falls_back_to_the_next_node_when_one_fails_lint(monkeypatch) -> None:
    ctx = _ctx_for_loop()
    selection = Selection(chosen="add_deny_rule", witness=_witness("a"), fallbacks=[("remove_endpoint", _witness("b"))])
    shipped = DefenderOutput(ok=True, new_policy_yaml="y", resource_type="endpoint_removal")
    attempts: list[str] = []

    def ship(_ctx, node_name, witness):
        attempts.append(node_name)
        if node_name == "add_deny_rule":
            raise LintViolationError("widens access")
        return shipped

    monkeypatch.setattr(agent, "_resolve_selection", lambda _ctx: selection)
    monkeypatch.setattr(agent, "_synthesize_and_ship", ship)

    assert agent._run_selection_loop(ctx) is shipped
    assert attempts == ["add_deny_rule", "remove_endpoint"]


def test_selection_loop_never_ships_a_later_node_after_the_first_succeeds(monkeypatch) -> None:
    ctx = _ctx_for_loop()
    selection = Selection(chosen="add_deny_rule", witness=_witness("a"), fallbacks=[("remove_endpoint", _witness("b"))])
    attempts: list[str] = []

    def ship(_ctx, node_name, witness):
        attempts.append(node_name)
        return DefenderOutput(ok=True, resource_type="l7_deny_rule")

    monkeypatch.setattr(agent, "_resolve_selection", lambda _ctx: selection)
    monkeypatch.setattr(agent, "_synthesize_and_ship", ship)

    assert agent._run_selection_loop(ctx).resource_type == "l7_deny_rule"
    assert attempts == ["add_deny_rule"]


@pytest.mark.parametrize("error", [LintViolationError("lint"), ValueError("bad"), KeyError("k")])
def test_selection_loop_abstains_with_the_last_error_when_every_node_fails(monkeypatch, error) -> None:
    ctx = _ctx_for_loop()
    selection = Selection(chosen="add_deny_rule", witness=_witness("a"), fallbacks=[("remove_endpoint", _witness("b"))])

    def ship(*_args):
        raise error

    monkeypatch.setattr(agent, "_resolve_selection", lambda _ctx: selection)
    monkeypatch.setattr(agent, "_synthesize_and_ship", ship)

    output = agent._run_selection_loop(ctx)

    assert output.ok is False
    assert output.new_policy_yaml is None
    assert output.error_message.startswith("no_feasible_node: every candidate node failed synthesis or lint")


def test_selection_loop_abstains_when_nodes_decline_without_raising(monkeypatch) -> None:
    ctx = _ctx_for_loop()
    selection = Selection(chosen="add_deny_rule", witness=_witness("a"))
    monkeypatch.setattr(agent, "_resolve_selection", lambda _ctx: selection)
    monkeypatch.setattr(agent, "_synthesize_and_ship", lambda *_a: None)

    output = agent._run_selection_loop(ctx)

    assert output.error_message == "no_feasible_node: no candidate node produced a valid patch"


def test_selection_loop_does_not_swallow_unexpected_exception_types(monkeypatch) -> None:
    ctx = _ctx_for_loop()
    selection = Selection(chosen="add_deny_rule", witness=_witness("a"))

    def ship(*_args):
        raise RuntimeError("programming error")

    monkeypatch.setattr(agent, "_resolve_selection", lambda _ctx: selection)
    monkeypatch.setattr(agent, "_synthesize_and_ship", ship)

    with pytest.raises(RuntimeError):
        agent._run_selection_loop(ctx)


_UNINSPECTABLE_POLICY = """\
version: 1
network_policies:
  raw:
    name: raw
    endpoints:
    - {host: raw.example.com, port: 443, protocol: tcp, tls: skip}
"""


def test_no_selection_on_an_l4_passthrough_endpoint_is_reported_as_unobservable(monkeypatch) -> None:
    cut = RequestTuple(host="raw.example.com", port=443)
    ctx = _ctx_for_loop(_UNINSPECTABLE_POLICY, cut)
    monkeypatch.setattr(
        agent,
        "_resolve_selection",
        lambda _ctx: Selection(infeasible={"add_deny_rule": "no L7", "remove_endpoint": "x"}),
    )

    output = agent._run_selection_loop(ctx)

    assert output.error_message.startswith("unobservable: no mitigation node's preconditions were met")
    assert "add_deny_rule: no L7" in output.error_message  # per-node reasons are surfaced to the reviewer


@pytest.mark.parametrize(
    "cut",
    [None, RequestTuple(path="/x"), RequestTuple(host="missing.example.com", port=443)],
    ids=["no-cut", "no-host", "host-not-in-policy"],
)
def test_all_uninspectable_is_false_without_a_resolvable_endpoint(cut) -> None:
    assert agent._all_uninspectable(_ctx_for_loop(_UNINSPECTABLE_POLICY, cut)) is False


def test_all_uninspectable_is_false_for_an_inspectable_endpoint() -> None:
    cut = RequestTuple(host="api.github.com", port=443)

    assert agent._all_uninspectable(_ctx_for_loop(cut=cut)) is False
    assert agent._all_uninspectable(
        _ctx_for_loop(_UNINSPECTABLE_POLICY, RequestTuple(host="raw.example.com", port=443))
    )


def test_abstain_output_without_per_node_reasons_keeps_the_detail_verbatim() -> None:
    output = agent._abstain_output(_ctx_for_loop(), AbstainReason(code="out_of_scope", detail="nothing to do"))

    assert output.error_message == "out_of_scope: nothing to do"
    assert output.resource_type is None
