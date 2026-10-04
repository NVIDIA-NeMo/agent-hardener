# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the gap_detector subgraph's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import GapDetectorState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def _interview_open(parent: SynthState) -> bool:
    """Whether another interview round can still run.

    Gates the per-tool boundary LLM fan-out: on the closing pass (round or
    question budget exhausted, or non-interactive) there is no way to collect
    answers, so the boundary calls would be pure waste.
    """
    return (
        parent.interactive
        and parent.interview_rounds < parent.max_interview_rounds
        and parent.interview_questions_asked < parent.max_interview_questions
    )


def entry_adapter(parent: SynthState) -> GapDetectorState:
    """Pull the merged profile, partials, and the boundary-pass inputs."""
    d = defaults()
    return GapDetectorState(
        profile=parent.profile,
        partials=dict(parent.partials),
        interviewer_answers=list(parent.interviewer_answers),
        interview_open=_interview_open(parent),
        concurrency=d.synth_llm.concurrency,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: GapDetectorState) -> dict[str, object]:
    """Surface only ``gaps`` back to the parent state."""
    return {"gaps": sub.gaps}
