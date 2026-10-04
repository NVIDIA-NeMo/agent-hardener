# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Consolidate a run's per-item validation results into a single ``validation.json`` artifact.

The attack-replay and benign validators already emit rich per-item verdicts inside each iteration's
``ValidatorReport.metadata``. This flattens the final iteration's reports into one run-level document a
downstream consumer (the NeMo plugin / Studio) can render as a sanity-check scorecard: which recorded
attacks are now blocked, and which benign requests are wrongly blocked (false positives).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.final_log.helpers import reversed_iterations

if TYPE_CHECKING:
    from agent_hardener.models.reports import RoundIterationReport, RoundReport


def _final_iteration(reports: list[RoundReport]) -> RoundIterationReport | None:
    """The last iteration that ran validators — reflects the final validated victim state."""
    return next((iteration for iteration in reversed_iterations(reports) if iteration.validators), None)


def _attack_row(result: dict[str, Any]) -> dict[str, Any]:
    verdict = result.get("verdict") or {}
    return {
        "attack_id": result.get("attack_id"),
        "probe": result.get("probe"),
        "goal": result.get("goal"),
        "prompt_excerpt": result.get("prompt_excerpt"),
        "status": verdict.get("status"),  # "blocked" | "not_blocked" | "error"
        "confidence": verdict.get("confidence"),
    }


def _benign_row(result: dict[str, Any]) -> dict[str, Any]:
    verdict = result.get("verdict") or {}
    raw = verdict.get("status")  # "complied" | "refused" | "error"
    return {
        "index": result.get("index"),
        "tool": result.get("tool"),
        "label": result.get("label"),
        "persona": result.get("persona"),
        "payload_excerpt": result.get("payload_excerpt"),
        # A complied benign request passed; a refused one is a false positive (legit call wrongly blocked).
        "status": "passed" if raw == "complied" else raw,
        "confidence": verdict.get("confidence"),
    }


def build_validation(reports: list[RoundReport]) -> dict[str, Any]:
    """Flatten the final iteration's attack + benign validator reports into one results document.

    Returns ``{}`` when no validators ran (nothing to persist).
    """
    iteration = _final_iteration(reports)
    if iteration is None:
        return {}

    attacks: list[dict[str, Any]] = []
    benign: list[dict[str, Any]] = []
    for report in iteration.validators:
        if report.kind == "attack":
            attacks.extend(_attack_row(r) for r in report.metadata.get("attack_results", []))
        elif report.kind == "benign":
            benign.extend(_benign_row(r) for r in report.metadata.get("results", []))

    if not attacks and not benign:
        return {}

    return {
        "attacks": attacks,
        "benign": benign,
        "summary": {
            "attacks_total": len(attacks),
            "attacks_blocked": sum(1 for a in attacks if a["status"] == "blocked"),
            "benign_total": len(benign),
            "benign_false_positives": sum(1 for b in benign if b["status"] == "refused"),
            # Reported alongside, never folded into the line above: a row the judge could not score
            # is not evidence that the guardrail let it through. Counting it as "not a false
            # positive" is the flattering reading, and it is how a run once reported one false
            # positive where three legitimate requests had in fact been refused.
            "benign_unscored": sum(1 for b in benign if b["status"] == "error"),
        },
    }
