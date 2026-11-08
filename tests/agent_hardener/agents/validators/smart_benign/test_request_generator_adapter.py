# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for request_generator/adapters.py entry_adapter borderline gating."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator import adapters
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.adapters import entry_adapter
from agent_hardener.agents.validators.smart_benign.subgraphs.request_generator.state import RequestGeneratorState

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _state(*, interactive: bool) -> SynthState:
    profile = VictimCapabilityProfile(
        target_name="test-agent",
        system_role="role",
        tools=[ToolSpec(name="bash_executor", description="runs bash", source="nl", confidence=0.9)],
        personas=[],
        out_of_scope=[],
        input_hash="abc123",
        generated_at=_NOW,
    )
    return SynthState(inputs=SynthInputs(target_name="test-agent"), interactive=interactive, profile=profile)


def test_borderline_off_by_default() -> None:
    assert defaults().generation.borderline_per_tool == 0
    assert entry_adapter(_state(interactive=True)).borderline_per_tool == 0


def test_negative_controls_off_by_default() -> None:
    assert defaults().generation.negative_control_per_tool == 0
    assert entry_adapter(_state(interactive=True)).negative_control_per_tool == 0
    assert RequestGeneratorState.model_fields["negative_control_per_tool"].default == 0


@pytest.fixture
def borderline_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    configured = defaults()
    configured.generation.borderline_per_tool = 2
    monkeypatch.setattr(adapters, "defaults", lambda: configured)


@pytest.mark.usefixtures("borderline_configured")
def test_borderline_enabled_when_interactive() -> None:
    assert entry_adapter(_state(interactive=True)).borderline_per_tool == 2


@pytest.mark.usefixtures("borderline_configured")
def test_borderline_suppressed_when_non_interactive_even_if_configured() -> None:
    assert entry_adapter(_state(interactive=False)).borderline_per_tool == 0


def test_borderline_suppressed_when_non_interactive() -> None:
    sub = entry_adapter(_state(interactive=False))
    assert sub.borderline_per_tool == 0
