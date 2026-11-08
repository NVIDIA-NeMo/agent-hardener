# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from agent_hardener.final_log.validation_summary import build_validation
from agent_hardener.runtime.stages.defense import DefenseStage


def test_build_validation_flattens_attack_and_benign() -> None:
    iteration = SimpleNamespace(
        validators=[
            SimpleNamespace(
                kind="attack",
                metadata={
                    "attack_results": [
                        {
                            "attack_id": "a1",
                            "probe": "p",
                            "goal": "g",
                            "prompt_excerpt": "x",
                            "verdict": {"status": "blocked", "confidence": 0.9},
                        },
                        {"attack_id": "a2", "probe": "p", "verdict": {"status": "not_blocked", "confidence": 0.8}},
                    ]
                },
            ),
            SimpleNamespace(
                kind="benign",
                metadata={
                    "results": [
                        {
                            "index": 1,
                            "tool": "clock",
                            "label": "b",
                            "persona": "",
                            "payload_excerpt": "now",
                            "verdict": {"status": "complied", "confidence": 0.9},
                        },
                        {"index": 2, "tool": "mail", "verdict": {"status": "refused", "confidence": 0.7}},
                        {"index": 3, "tool": "mail", "verdict": {"status": "error", "confidence": 0.0}},
                    ]
                },
            ),
        ]
    )
    report = SimpleNamespace(iterations=[iteration])

    out = build_validation([report])

    assert out["summary"] == {
        "attacks_total": 2,
        "attacks_blocked": 1,
        "benign_total": 3,
        "benign_false_positives": 1,
        # Counted separately, never as "not a false positive": a row the judge could not score is
        # no evidence the guardrail let it through, and hiding it makes the run read better than it was.
        "benign_unscored": 1,
    }
    assert out["attacks"][0]["status"] == "blocked"
    assert out["benign"][0]["status"] == "passed"  # complied -> passed
    assert out["benign"][1]["status"] == "refused"  # refused -> false positive


def test_build_validation_empty_without_validators() -> None:
    assert build_validation([SimpleNamespace(iterations=[])]) == {}


def test_defense_stage_no_defenders_is_noop() -> None:
    # A frozen validate-only run configures zero defenders; the stage must no-op without touching ctx
    # (and without a router LLM call).
    stage = DefenseStage(SimpleNamespace(defenders=[]), SimpleNamespace())
    result = asyncio.run(stage.run(ctx=None))
    assert result.analyses == []
    assert result.policy_patches == []


def _facts(target_called: bool, ran: bool, **extra: object) -> dict:
    return {
        "trace_status": "inspected",
        "target_tool": "x",
        "target_tool_called": target_called,
        "target_tool_outcome": "ok" if ran else "not_called",
        "target_tool_ran": ran,
        "blocked_tool": None,
        "rail": None,
        **extra,
    }


def test_build_validation_with_traces_counts_relabels_and_tool_facts() -> None:
    """With traces the rows show the final status and the tool facts, and the summary gains trace counts."""
    attack_results = [
        {"attack_id": "ran", "verdict": {"status": "not_blocked"}, "tool_facts": _facts(True, True)},
        {
            "attack_id": "rail",
            "verdict": {"status": "not_blocked"},
            "trace_verdict": {"status": "blocked_guardrail"},
            "tool_facts": _facts(True, False, blocked_tool="x", rail="r1"),
        },
        {"attack_id": "detector-blocked-but-ran", "verdict": {"status": "blocked"}, "tool_facts": _facts(True, True)},
        {"attack_id": "untraced", "verdict": {"status": "not_blocked"}, "tool_facts": {"trace_status": "untraced"}},
    ]
    benign_results = [
        {"index": 1, "verdict": {"status": "complied"}, "tool_facts": _facts(False, False)},
        {
            "index": 2,
            "verdict": {"status": "refused"},
            "trace_verdict": {"status": "blocked_network"},
            "tool_facts": _facts(True, False),
        },
        {
            "index": 3,
            "verdict": {"status": "refused"},
            "trace_verdict": {"status": "agent_refused"},
            "tool_facts": _facts(True, True),
        },
        {"index": 4, "verdict": {"status": "refused"}, "tool_facts": {"trace_status": "untraced"}},
    ]
    iteration = SimpleNamespace(
        validators=[
            SimpleNamespace(kind="attack", metadata={"attack_results": attack_results}),
            SimpleNamespace(kind="benign", metadata={"results": benign_results}),
        ]
    )

    out = build_validation([SimpleNamespace(iterations=[iteration])])

    assert [a["status"] for a in out["attacks"]] == ["not_blocked", "blocked_guardrail", "blocked", "not_blocked"]
    assert (out["attacks"][1]["blocked_tool"], out["attacks"][1]["rail"]) == ("x", "r1")
    assert out["attacks"][3]["trace_status"] == "untraced"
    assert [b["status"] for b in out["benign"]] == ["passed", "blocked_network", "agent_refused", "refused"]
    assert not any("_facts" in row for row in [*out["attacks"], *out["benign"]])
    summary = out["summary"]
    assert (summary["attacks_blocked"], summary["attacks_blocked_guardrail"], summary["attacks_traced"]) == (2, 1, 3)
    assert (summary["attacks_target_tool_called"], summary["attacks_blocked_but_tool_ran"]) == (3, 1)
    assert (summary["benign_false_positives"], summary["benign_blocked_network"], summary["benign_agent_refused"]) == (
        2,
        1,
        1,
    )
    assert (
        summary["benign_traced"],
        summary["benign_target_tool_called"],
        summary["benign_complied_without_tool"],
    ) == (
        3,
        2,
        1,
    )
