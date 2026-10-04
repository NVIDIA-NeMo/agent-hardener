# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for gap_detector/nodes/detect_gaps.py."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agent_hardener.agents.validators.smart_benign.models import ToolMergeProposal, ToolSpec, VictimCapabilityProfile
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.nodes.detect_gaps import (
    COMPLETENESS_GAP,
    _completeness_gaps,
    _dropped_tool_gaps,
    _merge_proposal_gaps,
    detect_gaps,
)
from agent_hardener.agents.validators.smart_benign.subgraphs.gap_detector.state import GapDetectorState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _tool(name: str, *, source: str = "nl", confidence: float = 0.9, description: str = "does things") -> ToolSpec:
    return ToolSpec(name=name, description=description, source=source, confidence=confidence, example_inputs=["hi"])


def _profile(
    tools: list[ToolSpec] | None = None,
    proposed_tool_merges: list[ToolMergeProposal] | None = None,
) -> VictimCapabilityProfile:
    return VictimCapabilityProfile(
        target_name="test-agent",
        system_role="test role",
        tools=tools or [],
        personas=[],
        out_of_scope=["nothing"],
        proposed_tool_merges=proposed_tool_merges or [],
        input_hash="abc123",
        generated_at=_NOW,
    )


# ---------------------------------------------------------------------------
# _dropped_tool_gaps
# ---------------------------------------------------------------------------


def test_dropped_tool_gaps_empty_partials() -> None:
    assert _dropped_tool_gaps([], {}) == []


def test_dropped_tool_gaps_all_present() -> None:
    tools = [_tool("bash_executor")]
    partials = {"nl": [_tool("bash_executor")]}
    assert _dropped_tool_gaps(tools, partials) == []


def test_dropped_tool_gaps_detects_missing() -> None:
    profile_tools = [_tool("bash_executor")]
    partials = {"nl": [_tool("bash_executor"), _tool("file_reader")]}
    gaps = _dropped_tool_gaps(profile_tools, partials)
    assert len(gaps) == 1
    assert "file_reader" in gaps[0]
    assert "nl" in gaps[0]


def test_dropped_tool_gaps_multiple_sources() -> None:
    profile_tools = [_tool("bash_executor")]
    partials = {
        "nl": [_tool("bash_executor")],
        "github": [_tool("bash_executor"), _tool("web_search")],
        "api_probe": [_tool("missing_tool")],
    }
    gaps = _dropped_tool_gaps(profile_tools, partials)
    names_in_gaps = " ".join(gaps)
    assert "web_search" in names_in_gaps
    assert "missing_tool" in names_in_gaps
    assert "bash_executor" not in names_in_gaps


# ---------------------------------------------------------------------------
# _completeness_gaps
# ---------------------------------------------------------------------------


def test_completeness_gap_emitted_when_tools_present() -> None:
    profile = _profile(tools=[_tool("bash_executor")])
    gaps = _completeness_gaps(profile)
    assert gaps == [COMPLETENESS_GAP]


def test_completeness_gap_suppressed_when_no_tools() -> None:
    profile = _profile(tools=[])
    assert _completeness_gaps(profile) == []


# ---------------------------------------------------------------------------
# _merge_proposal_gaps
# ---------------------------------------------------------------------------


def test_merge_proposal_gap_emitted_per_proposal() -> None:
    proposals = [
        ToolMergeProposal(canonical_name="bash_executor", aliases=["shell_executor"], rationale="same capability"),
    ]
    profile = _profile(tools=[_tool("bash_executor"), _tool("shell_executor")], proposed_tool_merges=proposals)
    gaps = _merge_proposal_gaps(profile)
    assert len(gaps) == 1
    assert "bash_executor" in gaps[0]
    assert "shell_executor" in gaps[0]


def test_merge_proposal_gap_empty_when_no_proposals() -> None:
    profile = _profile(tools=[_tool("bash_executor")])
    assert _merge_proposal_gaps(profile) == []


def test_merge_proposal_gap_multiple_proposals() -> None:
    proposals = [
        ToolMergeProposal(canonical_name="bash_executor", aliases=["shell_executor"], rationale="a"),
        ToolMergeProposal(canonical_name="python_executor", aliases=["code_runner"], rationale="b"),
    ]
    profile = _profile(tools=[], proposed_tool_merges=proposals)
    gaps = _merge_proposal_gaps(profile)
    assert len(gaps) == 2


# ---------------------------------------------------------------------------
# detect_gaps (integration — async)
# ---------------------------------------------------------------------------


def test_detect_gaps_includes_completeness_when_tools_present() -> None:
    state = GapDetectorState(
        profile=_profile(
            tools=[
                ToolSpec(
                    name="bash_executor",
                    description="runs bash",
                    source="nl",
                    confidence=0.9,
                    example_inputs=["run ls"],
                )
            ]
        ),
        partials={},
    )
    result = asyncio.run(detect_gaps(state))
    assert COMPLETENESS_GAP in result["gaps"]


def test_detect_gaps_no_completeness_when_no_tools() -> None:
    state = GapDetectorState(profile=_profile(tools=[]), partials={})
    result = asyncio.run(detect_gaps(state))
    assert COMPLETENESS_GAP not in result["gaps"]


def test_detect_gaps_includes_dropped_tool() -> None:
    dropped = _tool("dropped_tool", source="github")
    state = GapDetectorState(
        profile=_profile(
            tools=[
                ToolSpec(
                    name="bash_executor",
                    description="runs bash",
                    source="nl",
                    confidence=0.9,
                    example_inputs=["run ls"],
                )
            ]
        ),
        partials={"github": [dropped]},
    )
    result = asyncio.run(detect_gaps(state))
    gap_text = " ".join(result["gaps"])
    assert "dropped_tool" in gap_text


def test_detect_gaps_missing_profile() -> None:
    state = GapDetectorState(profile=None)
    result = asyncio.run(detect_gaps(state))
    assert len(result["gaps"]) == 1
    assert "profile is missing" in result["gaps"][0]
