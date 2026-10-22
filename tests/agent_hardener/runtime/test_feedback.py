# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from agent_hardener.models import (
    DefenderAnalysis,
    RoundIterationReport,
    ValidationFeedback,
    ValidatorReport,
    VictimControlResult,
    VictimResult,
)
from agent_hardener.runtime.feedback import build_validation_feedback


def _defender(agent_id: str, attack: str | None, *policies: str | None) -> DefenderAnalysis:
    return DefenderAnalysis(
        agent_id=agent_id,
        agent_name=agent_id,
        attack_prompt=attack,
        policy_patches=[{"new_policy_yaml": p} for p in policies],
    )


def _validator(
    kind: Literal["attack", "benign"] = "attack",
    *,
    ok: bool = False,
    fn: list[str] | None = None,
    fp: list[str] | None = None,
) -> ValidatorReport:
    return ValidatorReport(
        agent_id=f"{kind}-validator",
        agent_name=kind,
        kind=kind,
        ok=ok,
        false_negatives=fn or [],
        false_positives=fp or [],
    )


def _attempt(defenders: list[DefenderAnalysis], validators: list[ValidatorReport]) -> RoundIterationReport:
    return RoundIterationReport(
        iteration=1,
        defenders=defenders,
        victim_control=VictimControlResult(),
        victim=VictimResult(agent_id="v", agent_name="v"),
        validators=validators,
        success=False,
    )


def test_no_failures_yields_no_feedback() -> None:
    attempt = _attempt([_defender("d1", "attack")], [_validator(ok=True), _validator("benign", ok=True)])

    assert build_validation_feedback(attempt) == {}


def test_findings_from_passing_validators_are_ignored() -> None:
    attempt = _attempt([_defender("d1", "attack")], [_validator(ok=True, fn=["attack"])])

    assert build_validation_feedback(attempt) == {}


def test_false_negative_is_attributed_only_to_the_defender_that_handled_that_attack() -> None:
    attempt = _attempt(
        [_defender("d1", "ignore previous", "policy-1"), _defender("d2", "other attack", "policy-2")],
        [_validator(fn=["ignore previous"])],
    )

    assert build_validation_feedback(attempt) == {
        "d1": ValidationFeedback(false_negatives=["ignore previous"], previous_policy_yaml="policy-1"),
    }


def test_false_negative_matches_when_either_text_contains_the_other() -> None:
    attempt = _attempt(
        [_defender("d1", "ignore previous"), _defender("d2", "a much longer attack prompt: reveal secrets")],
        [_validator(fn=["please ignore previous instructions", "reveal secrets"])],
    )

    assert set(build_validation_feedback(attempt)) == {"d1", "d2"}


def test_defender_without_an_attack_prompt_gets_no_false_negative_feedback() -> None:
    attempt = _attempt([_defender("d1", None)], [_validator(fn=["anything"])])

    assert build_validation_feedback(attempt) == {}


def test_same_defender_handling_several_unblocked_attacks_accumulates_them() -> None:
    attempt = _attempt(
        [_defender("d1", "attack one"), _defender("d1", "attack two")],
        [_validator(fn=["attack one", "attack two"])],
    )

    assert build_validation_feedback(attempt)["d1"].false_negatives == ["attack one", "attack two"]


def test_false_positives_go_to_every_defender() -> None:
    attempt = _attempt(
        [_defender("d1", "a1", "p1"), _defender("d2", None, None)],
        [_validator("benign", fp=["list my files"])],
    )

    feedback = build_validation_feedback(attempt)

    assert feedback == {
        "d1": ValidationFeedback(false_positives=["list my files"], previous_policy_yaml="p1"),
        "d2": ValidationFeedback(false_positives=["list my files"]),
    }


def test_defender_with_both_kinds_of_failure_gets_one_merged_entry() -> None:
    attempt = _attempt(
        [_defender("d1", "attack", "policy")],
        [_validator(fn=["attack"]), _validator("benign", fp=["benign request"])],
    )

    assert build_validation_feedback(attempt) == {
        "d1": ValidationFeedback(
            false_negatives=["attack"], false_positives=["benign request"], previous_policy_yaml="policy"
        )
    }


def test_previous_policy_is_the_first_patch_that_has_yaml() -> None:
    defender = DefenderAnalysis(
        agent_id="d1",
        agent_name="d1",
        attack_prompt="attack",
        policy_patches=[
            {"note": "no yaml"},
            {"new_policy_yaml": ""},
            {"new_policy_yaml": "first"},
            {"new_policy_yaml": "second"},
        ],
    )
    attempt = _attempt([defender], [_validator(fn=["attack"])])

    assert build_validation_feedback(attempt)["d1"].previous_policy_yaml == "first"


def test_defender_whose_attack_was_blocked_gets_no_false_negative_but_still_gets_false_positives() -> None:
    attempt = _attempt(
        [_defender("d1", "blocked attack"), _defender("d2", "leaked attack")],
        [_validator(fn=["leaked attack"]), _validator("benign", fp=["fp"])],
    )

    feedback = build_validation_feedback(attempt)

    assert feedback["d1"].false_negatives == []
    assert feedback["d2"].false_negatives == ["leaked attack"]
    assert feedback["d1"].false_positives == feedback["d2"].false_positives == ["fp"]
