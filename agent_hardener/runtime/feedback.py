# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Between-iterations transform: derive per-defender feedback from a failed iteration.

This is not a pipeline stage — it is the pure function that turns one iteration's validator reports
into the ``ValidationFeedback`` the next iteration's defenders manager consumes, so it lives in
``runtime/`` rather than ``runtime/stages/``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.models import ValidationFeedback

if TYPE_CHECKING:
    from agent_hardener.models import RoundIterationReport


def build_validation_feedback(attempt: RoundIterationReport) -> dict[str, ValidationFeedback]:
    """Build per-defender feedback from a failed attempt's validator reports.

    Attack validators supply false_negatives attributed to the defender that handled that attack.
    Benign validators supply false_positives attributed to all defenders in the attempt.
    """
    all_false_negatives = [fn for v in attempt.validators if not v.ok for fn in v.false_negatives]
    all_false_positives = [fp for v in attempt.validators if not v.ok for fp in v.false_positives]

    if not all_false_negatives and not all_false_positives:
        return {}

    feedback: dict[str, ValidationFeedback] = {}

    # False negatives: attribute to the defender whose attack_prompt matches.
    if all_false_negatives:
        for analysis in attempt.defenders:
            if not analysis.attack_prompt:
                continue
            # Check if this defender's attack was among the unblocked ones.
            if not any(analysis.attack_prompt in fn or fn in analysis.attack_prompt for fn in all_false_negatives):
                continue
            previous_policy = next(
                (p.get("new_policy_yaml") for p in analysis.policy_patches if p.get("new_policy_yaml")),
                None,
            )
            existing = feedback.get(analysis.agent_id)
            if existing:
                feedback[analysis.agent_id] = existing.model_copy(
                    update={"false_negatives": [*existing.false_negatives, analysis.attack_prompt]}
                )
            else:
                feedback[analysis.agent_id] = ValidationFeedback(
                    false_negatives=[analysis.attack_prompt],
                    false_positives=[],
                    previous_policy_yaml=previous_policy,
                )

    # False positives: attribute to all defenders (any guardrail could be the cause).
    if all_false_positives:
        for analysis in attempt.defenders:
            previous_policy = next(
                (p.get("new_policy_yaml") for p in analysis.policy_patches if p.get("new_policy_yaml")),
                None,
            )
            existing = feedback.get(analysis.agent_id)
            if existing:
                feedback[analysis.agent_id] = existing.model_copy(update={"false_positives": all_false_positives})
            else:
                feedback[analysis.agent_id] = ValidationFeedback(
                    false_negatives=[],
                    false_positives=all_false_positives,
                    previous_policy_yaml=previous_policy,
                )

    return feedback
