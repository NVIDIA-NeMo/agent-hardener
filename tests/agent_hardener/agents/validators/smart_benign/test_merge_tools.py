# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Node tests for the deterministic tool merge in the profile_synthesizer subgraph."""

from __future__ import annotations

import asyncio

import pytest

from agent_hardener.agents.validators.smart_benign.models import ToolSource, ToolSpec
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.nodes import merge as mg
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_synthesizer.state import ProfileSynthesizerState


def _tool(name: str, source: ToolSource, *examples: str, description: str | None = None) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or f"{name} from {source}",
        example_inputs=list(examples),
        source=source,
        confidence=0.5,
    )


def _state(partials: dict[str, list[ToolSpec]]) -> ProfileSynthesizerState:
    return ProfileSynthesizerState(
        target_name="victim",
        partials=partials,
        input_hash="h",
        synth_model="m",
        synth_base_url="http://llm.test",
        synth_api_key="k",
    )


def _merged(partials: dict[str, list[ToolSpec]]) -> list[ToolSpec]:
    delta = asyncio.run(mg.merge(_state(partials)))
    merged = delta["merged_tools"]
    assert isinstance(merged, list)
    return merged


def test_no_partials_merge_to_nothing() -> None:
    assert asyncio.run(mg.merge(_state({}))) == {"merged_tools": []}


def test_distinct_tools_from_different_sources_are_all_kept() -> None:
    merged = _merged({"nl": [_tool("bash", "nl")], "github": [_tool("search", "github")]})

    assert sorted(t.name for t in merged) == ["bash", "search"]


@pytest.mark.parametrize(
    ("sources", "winner"),
    [
        (["nl", "github"], "github"),
        (["api_probe", "nl"], "nl"),
        (["interview", "api_probe"], "api_probe"),
        (["interview", "api_probe", "nl", "github"], "github"),
        (["github", "interview"], "github"),
    ],
)
def test_the_highest_priority_source_wins_regardless_of_input_order(sources: list[ToolSource], winner: str) -> None:
    partials: dict[str, list[ToolSpec]] = {source: [_tool("bash", source)] for source in sources}

    (tool,) = _merged(partials)

    assert tool.source == winner
    assert tool.description == f"bash from {winner}"


def test_examples_are_unioned_in_first_seen_order_without_duplicates() -> None:
    partials = {
        "nl": [_tool("bash", "nl", "ls", "pwd")],
        "github": [_tool("bash", "github", "pwd", "whoami")],
        "api_probe": [_tool("bash", "api_probe", "ls", "date")],
    }

    (tool,) = _merged(partials)

    assert tool.source == "github"
    assert tool.example_inputs == ["ls", "pwd", "whoami", "date"]


def test_the_winner_is_copied_not_mutated() -> None:
    original = _tool("bash", "github", "a")
    partials = {"github": [original], "nl": [_tool("bash", "nl", "b")]}

    (tool,) = _merged(partials)

    assert tool.example_inputs == ["a", "b"]
    assert original.example_inputs == ["a"]


def test_a_tool_listed_twice_by_one_source_still_merges_to_one() -> None:
    (tool,) = _merged({"nl": [_tool("bash", "nl", "a"), _tool("bash", "nl", "b")]})

    assert tool.example_inputs == ["a", "b"]
