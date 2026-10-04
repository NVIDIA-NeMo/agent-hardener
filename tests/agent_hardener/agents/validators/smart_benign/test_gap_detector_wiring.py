# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gap_detector adapters and agent factory."""

from __future__ import annotations

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector import (
    build_gap_detector_agent,
    entry_adapter,
    exit_adapter,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.state import GapDetectorState


def test_entry_adapter_carries_answers_and_opens_interview() -> None:
    parent = SynthState(
        inputs=SynthInputs(target_name="t"),
        interactive=True,
        interviewer_answers=[("boundary[x]: q", "Q?", "a")],
    )
    sub = entry_adapter(parent)
    assert sub.interviewer_answers == [("boundary[x]: q", "Q?", "a")]
    assert sub.synth_model  # populated from defaults
    # Fresh interview (rounds/questions unspent) → boundary pass is open.
    assert sub.interview_open is True
    # Concurrency is the synth fan-out budget, not the replay one.
    assert sub.concurrency == defaults().synth_llm.concurrency


def test_entry_adapter_closes_interview_when_rounds_exhausted() -> None:
    parent = SynthState(
        inputs=SynthInputs(target_name="t"),
        interactive=True,
        max_interview_rounds=2,
        interview_rounds=2,
    )
    sub = entry_adapter(parent)
    assert sub.interview_open is False


def test_entry_adapter_closes_interview_when_non_interactive() -> None:
    parent = SynthState(inputs=SynthInputs(target_name="t"), interactive=False)
    assert entry_adapter(parent).interview_open is False


def test_exit_adapter_surfaces_gaps() -> None:
    sub = GapDetectorState(gaps=["a", "boundary[x]: q"])
    assert exit_adapter(sub) == {"gaps": ["a", "boundary[x]: q"]}


def test_build_agent_compiles() -> None:
    agent = build_gap_detector_agent()
    assert agent is not None
