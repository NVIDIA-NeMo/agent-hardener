# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the agent display registry and phase summary renderers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.display import (
    DisplayContext,
    render_attack_summary,
    render_defender_summary,
    render_outputs_verbose,
    render_validator_summary,
    resolve_renderer,
)
from agent_hardener.display.renderers.garak_attacker import GarakAttackerRenderer
from agent_hardener.display.renderers.generic import GenericRenderer
from agent_hardener.display.renderers.openshell_policy import PolicyDefenderRenderer
from agent_hardener.display.renderers.smart_benign import SmartBenignValidatorRenderer
from agent_hardener.models import Artifact

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_resolve_renderer_by_implementation_prefix() -> None:
    output = {
        "agent_name": "agent-breaker",
        "implementation": "agent_hardener.agents.attackers.agent_breaker:run",
        "records": [],
    }
    assert isinstance(resolve_renderer(output), GarakAttackerRenderer)


def test_resolve_renderer_by_display_artifact_type() -> None:
    output = {
        "agent_name": "custom",
        "artifacts": [{"type": "benign.requests_csv", "path": "/tmp/requests.csv"}],
    }
    assert isinstance(resolve_renderer(output), SmartBenignValidatorRenderer)


def test_resolve_renderer_defender_role() -> None:
    output = {
        "agent_name": "guardrails",
        "policy_patches": [],
        "metadata": {"workflow": "guardrails_defender"},
    }
    assert isinstance(resolve_renderer(output), PolicyDefenderRenderer)


def test_resolve_renderer_generic_fallback() -> None:
    output = {"agent_name": "stub", "summary": "done"}
    assert isinstance(resolve_renderer(output), GenericRenderer)


def test_render_attack_summary_headline_and_verbose_table() -> None:
    line, verbose = render_attack_summary(
        {"attacks": [{"agent_name": "agent_breaker", "records": [{"probe": "p"}] * 2}]},
        DisplayContext(verbose=True),
    )
    assert "2 hits" in str(line)
    assert verbose


def test_render_defender_summary_headlines() -> None:
    lines, verbose = render_defender_summary(
        {
            "defenders": [
                {"agent_name": "policy", "ok": True, "policy_patches": [{}]},
                {"agent_name": "guardrails", "ok": False, "policy_patches": []},
            ],
            "policy_patches": [{}],
        },
        DisplayContext(verbose=False),
    )
    assert len(lines) == 2
    assert not verbose


def test_render_validator_summary_and_generic_verbose(tmp_path: Path) -> None:
    artifact = tmp_path / "notes.txt"
    artifact.write_text("x\n", encoding="utf-8")
    lines, verbose = render_validator_summary(
        {
            "validators": [
                {
                    "agent_name": "custom-validator",
                    "ok": True,
                    "summary": "all good",
                    "artifacts": [{"type": "text.plain", "path": str(artifact), "label": "notes"}],
                }
            ]
        },
        DisplayContext(verbose=True),
    )
    assert len(lines) == 1
    assert verbose
    assert "notes" in str(verbose[0])


def test_render_outputs_verbose_uses_registry(tmp_path: Path) -> None:
    csv_path = tmp_path / "requests.csv"
    csv_path.write_text("tool,payload\n", encoding="utf-8")
    output = {
        "agent_name": "smart-benign",
        "kind": "benign",
        "summary": "0/1 complied",
        "metadata": {"results": [{"tool": "bash", "verdict": {"status": "complied"}}]},
        "artifacts": [Artifact(type="benign.requests_csv", path=csv_path).model_dump(mode="json")],
    }
    parts = render_outputs_verbose([output], DisplayContext(verbose=True))
    assert parts
    assert "bash" in str(parts[0])
