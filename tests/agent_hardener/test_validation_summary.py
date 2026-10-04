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
