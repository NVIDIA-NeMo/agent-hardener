# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OpenShell Defender v2 — deterministic feasibility/selector network-policy defender.

``run()`` is stateless and re-entrant: it produces at most one candidate patch per call. Whether
that candidate is actually safe against real traffic is validated *downstream*, by the existing
pipeline validators, and fed back via ``DefenderInput.feedback`` on a subsequent call — this
agent never loops internally.

Scope: ``network_policies``/``network_middlewares`` only (OpenShell's dynamic, hot-reloadable
sections). ``filesystem_policy``/``landlock``/``process`` are static and out of scope — if the
harm can only be closed there, this agent abstains with ``out_of_scope`` rather than force a
patch it isn't allowed to make.
"""

from __future__ import annotations

import logging

from agent_hardener.env import inference_api_key
from agent_hardener.models.contracts import DefenderInput, DefenderOutput

from .analysis.benign_index import build_index, merge_counterexamples
from .analysis.overlap import compute_overlap
from .analysis.selector import select
from .config import DefenderConfig, load_config
from .errors import DefenderError, ExtractionError
from .extraction.attack import build_harm_certificate, extract_attack_tuples, select_cut
from .extraction.benign import predict_benign_tuples
from .extraction.environment import endpoint_is_protected, extract_environment
from .feedback import interpret
from .models import AbstainReason, Candidate, DefenderContext, Selection
from .nodes import get_node
from .policy.lint import assert_contraction
from .policy.loader import current_policy_text, dump_policy_yaml, load_policy_from_yaml
from .policy.patch import PolicyDelta, apply_delta
from .policy.query import endpoint_is_inspectable, find_endpoint, find_entry_for
from .telemetry import log_candidate, log_selection

logger = logging.getLogger(__name__)


def run(defender_input: DefenderInput) -> DefenderOutput:
    if not inference_api_key():
        raise ValueError("INFERENCE_API_KEY (or configured api_key_env) is not set")
    config = load_config()
    try:
        ctx = _build_context(defender_input, config)
    except ExtractionError as exc:
        return _abstain_output(
            DefenderContext(
                defender_input=defender_input, policy=load_policy_from_yaml(current_policy_text(defender_input))
            ),
            AbstainReason(code="extraction_failed", detail=str(exc)),
        )
    except Exception as exc:
        return _error_output(exc, 0)

    if ctx.cut is not None and ctx.cut.host is None:
        return _abstain_output(
            ctx,
            AbstainReason(
                code="out_of_scope",
                detail="attack has no network signature; the fix likely lives in filesystem_policy/landlock/process",
            ),
        )

    if ctx.cut is not None and endpoint_is_protected(ctx.protected_endpoints, ctx.cut.host, ctx.cut.port):
        return _abstain_output(
            ctx,
            AbstainReason(
                code="protected_endpoint",
                detail="this endpoint is the victim agent's own required backend/LLM endpoint and must never be modified",
            ),
        )

    try:
        return _run_selection_loop(ctx)
    except Exception as exc:
        return _error_output(exc, ctx.iteration)


def _build_context(defender_input: DefenderInput, config: DefenderConfig) -> DefenderContext:
    policy_text = current_policy_text(defender_input)
    policy = load_policy_from_yaml(policy_text)

    attack_tuples = extract_attack_tuples(defender_input, policy, config)
    harm_cert = build_harm_certificate(defender_input, attack_tuples, config)
    cut = select_cut(attack_tuples, harm_cert)

    cache = defender_input.context.get("defender_extraction_cache")
    benign_predictions = predict_benign_tuples(defender_input.benign_requests, config, cache=cache)
    benign_tuples = [t for prediction in benign_predictions for t in prediction.tuples]
    benign_index = build_index(benign_predictions)

    environment_tuples = extract_environment(defender_input, config, cache=cache)
    protected_endpoints = {(t.host, t.port) for t in environment_tuples}

    directive = interpret(defender_input.feedback, defender_input, policy, config)
    # Counterexamples must be merged before overlap is computed, so a benign request the
    # validator saw wrongly blocked changes this round's feasibility verdicts, not merely crosses
    # one node off the excluded list.
    if directive.counterexample_tuples:
        merge_counterexamples(benign_index, directive.counterexample_tuples)
        benign_tuples = benign_tuples + directive.counterexample_tuples

    overlap = compute_overlap(cut, benign_index)

    return DefenderContext(
        defender_input=defender_input,
        policy=policy,
        attack_tuples=attack_tuples,
        cut=cut,
        harm_cert=harm_cert,
        benign_index=benign_index,
        benign_tuples=benign_tuples,
        benign_predictions=benign_predictions,
        overlap=overlap,
        feedback=directive,
        iteration=(len(defender_input.feedback.false_negatives) + len(defender_input.feedback.false_positives))
        if defender_input.feedback
        else 0,
        protected_endpoints=protected_endpoints,
    )


def _resolve_selection(ctx: DefenderContext) -> Selection:
    excluded = ctx.feedback.excluded_nodes if ctx.feedback else set()
    selection = select(ctx, excluded)
    log_selection(ctx, selection)
    return selection


def _run_selection_loop(ctx: DefenderContext) -> DefenderOutput:
    """Try the selector's chosen node, then its fallbacks in order.

    Stops at the first one that synthesizes a lint-clean candidate, or exhausts the chain (a
    node's own preconditions can still reject a witness the feasibility check approved — e.g. an
    edge case ``synthesize`` catches that ``check_*`` didn't — so falling through here is
    strictly more resilient than a single try).
    """
    selection = _resolve_selection(ctx)
    if selection.chosen is None:
        code = "unobservable" if _all_uninspectable(ctx) else "no_feasible_node"
        return _abstain_output(
            ctx,
            AbstainReason(
                code=code, detail="no mitigation node's preconditions were met", per_node_reasons=selection.infeasible
            ),
        )

    candidates = [(selection.chosen, selection.witness), *selection.fallbacks]
    last_error: Exception | None = None
    for node_name, witness in candidates:
        try:
            output = _synthesize_and_ship(ctx, node_name, witness)
            if output is not None:
                return output
        except (DefenderError, ValueError, KeyError) as exc:
            last_error = exc
            logger.info("openshell_defender_v2 node %s failed, trying next fallback: %s", node_name, exc)
            continue
    if last_error is not None:
        return _abstain_output(
            ctx,
            AbstainReason(
                code="no_feasible_node", detail=f"every candidate node failed synthesis or lint: {last_error}"
            ),
        )
    return _abstain_output(
        ctx, AbstainReason(code="no_feasible_node", detail="no candidate node produced a valid patch")
    )


def _all_uninspectable(ctx: DefenderContext) -> bool:
    if ctx.cut is None or ctx.cut.host is None:
        return False
    found = find_entry_for(ctx.policy, ctx.cut.host, ctx.cut.port, ctx.cut.binary)
    if found is None:
        return False
    _key, entry = found
    endpoint = find_endpoint(entry, ctx.cut.host, ctx.cut.port)
    return endpoint is not None and not endpoint_is_inspectable(endpoint)


def _synthesize_and_ship(ctx: DefenderContext, node_name: str, witness) -> DefenderOutput | None:
    candidate = _synthesize(ctx, node_name, witness)
    log_candidate(ctx, candidate)
    delta = PolicyDelta.model_validate(candidate.delta)
    after = apply_delta(ctx.policy, delta)
    assert_contraction(ctx.policy, after)  # raises LintViolationError -> caller tries the next fallback
    return _to_output(ctx, candidate, after)


def _synthesize(ctx: DefenderContext, node_name: str, witness) -> Candidate:
    node = get_node(node_name)
    return node.synthesize(ctx, witness)


def _to_output(ctx: DefenderContext, candidate, after_policy) -> DefenderOutput:
    return DefenderOutput(
        ok=True,
        new_policy_yaml=dump_policy_yaml(after_policy),
        iteration_count=ctx.iteration,
        resource_type=candidate.resource_type,
        guardrail_name=candidate.guardrail_name,
    )


def _abstain_output(ctx: DefenderContext, reason: AbstainReason) -> DefenderOutput:
    detail = reason.detail
    if reason.per_node_reasons:
        detail += " (" + "; ".join(f"{k}: {v}" for k, v in reason.per_node_reasons.items()) + ")"
    return DefenderOutput(
        ok=False,
        error_message=f"{reason.code}: {detail}",
        new_policy_yaml=None,
        iteration_count=ctx.iteration,
        resource_type=None,
    )


def _error_output(exc: Exception, iteration: int) -> DefenderOutput:
    return DefenderOutput(
        ok=False, error_message=str(exc), new_policy_yaml=None, iteration_count=iteration, resource_type=None
    )
