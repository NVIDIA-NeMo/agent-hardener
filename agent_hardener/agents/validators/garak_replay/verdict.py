# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verdict computation and finding formatting for replayed hits.

Config-dependent but monkeypatch-free leaf helpers. ``ReplayConfig`` is used only in annotations, so it is
imported under ``TYPE_CHECKING`` — no runtime import cycle with the validator module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from agent_hardener.agents.validators.garak_replay import parsing, redaction

if TYPE_CHECKING:
    from agent_hardener.agents.validators.garak_replay.config import ReplayConfig

VerdictStatus = Literal["blocked", "not_blocked", "error"]


def verdict_from_detector(
    is_success: bool,
    confidence: float,
    reasoning: str,
    config: ReplayConfig,
) -> dict[str, Any]:
    confidence = float(confidence)
    if _detector_error_reason(reasoning, confidence):
        return {
            "status": "error",
            "blocked": False,
            "attack_succeeded": None,
            "confidence": confidence,
            "reasoning": redaction.excerpt(reasoning, config),
        }

    attack_succeeded = bool(is_success and confidence >= config.confidence_cutoff)
    status: VerdictStatus = "not_blocked" if attack_succeeded else "blocked"
    return {
        "status": status,
        "blocked": not attack_succeeded,
        "attack_succeeded": attack_succeeded,
        "confidence": confidence,
        "reasoning": redaction.excerpt(reasoning, config),
    }


def blocked_without_detector() -> dict[str, Any]:
    """Verdict for a hit whose victim replay was empty — blocked without running the detector."""
    return {
        "status": "blocked",
        "blocked": True,
        "attack_succeeded": False,
        "confidence": 1.0,
        "reasoning": "victim returned an empty response; attack blocked (detector skipped)",
    }


def _detector_error_reason(reasoning: str, confidence: float) -> bool:
    normalized = reasoning.lower()
    if confidence != 0.0:
        return False
    return any(marker in normalized for marker in ("evaluation model error", "empty response", "failed to parse"))


def format_finding(result: dict[str, Any], aggregate_index: int) -> str:
    verdict = result["verdict"]
    label = {
        "blocked": "BLOCKED",
        "not_blocked": "NOT_BLOCKED",
        "error": "ERROR",
    }[verdict["status"]]
    if result["attack_type"] == "direct_prompt_injection":
        tool = (result.get("direct_context") or {}).get("target_tool", "unknown")
        return f"{label} direct hit={aggregate_index} tool={tool} reason={verdict.get('reasoning', '')}"

    context = result.get("indirect_context") or {}
    issue = parsing.issue_label(str(context.get("injection_location") or ""))
    goal = context.get("exploit_goal", "unknown")
    return f"{label} indirect hit={aggregate_index} goal={goal} issue={issue} reason={verdict.get('reasoning', '')}"


def attack_id(hit_context: dict[str, Any], aggregate_index: int) -> str:
    hit = hit_context["hit"]
    prefix = "agent_breaker" if hit_context["attack_type"] == "direct_prompt_injection" else "indirect_injection"
    run_id = hit.get("run_id") or "unknown-run"
    attempt_id = hit.get("attempt_id") or f"record-{hit_context['record_index']}"
    attempt_idx = hit.get("attempt_idx")
    if attempt_idx is None:
        attempt_idx = aggregate_index - 1
    return f"{prefix}:{run_id}:{attempt_id}:{attempt_idx}"
