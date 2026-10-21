# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v2 import graph as graph_module
from agent_hardener.models import DefenderOutput

from .fakes import graph_state

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.agents.defenders.guardrails_defender_v2.schemas import GraphState


def test_pipeline_is_a_straight_line_without_the_validation_loop() -> None:
    compiled = graph_module.build_defender_graph().get_graph()

    edges = {(edge.source, edge.target) for edge in compiled.edges}
    assert edges == {
        ("__start__", "init"),
        ("init", "vulnerability_analyzer"),
        ("vulnerability_analyzer", "guardrail_generator"),
        ("guardrail_generator", "component_writer"),
        ("component_writer", "__end__"),
    }


def test_each_stage_feeds_the_next_through_graph_state(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def analyzer(state: GraphState) -> dict[str, Any]:
        seen.append("analyzer")
        return {"vulnerability_analysis": "ANALYSIS"}

    def generator(state: GraphState) -> dict[str, Any]:
        seen.append("generator")
        return {"draft_instruction": f"guard for {state.vulnerability_analysis}"}

    def writer(state: GraphState) -> dict[str, Any]:
        seen.append("writer")
        return {"final_output": DefenderOutput(ok=True, new_policy_yaml=state.draft_instruction)}

    monkeypatch.setattr(graph_module, "vulnerability_analyzer_node", analyzer)
    monkeypatch.setattr(graph_module, "custom_guardrails_generator_node", generator)
    monkeypatch.setattr(graph_module, "component_writer_node", writer)

    final = graph_module.build_defender_graph().invoke(graph_state())

    assert seen == ["analyzer", "generator", "writer"]
    assert final["final_output"].new_policy_yaml == "guard for ANALYSIS"


@pytest.fixture
def plugins_file(tmp_path: Path) -> Path:
    path = tmp_path / "plugins.toml"
    path.write_text("version = 1\n", encoding="utf-8")
    return path


@pytest.mark.parametrize("exists", [True, False])
def test_init_keeps_an_existing_plugins_path_and_drops_a_missing_one(
    plugins_file: Path, tmp_path: Path, exists: bool
) -> None:
    path = plugins_file if exists else tmp_path / "absent.toml"
    state = graph_state({"relay_plugins_path": path})

    assert graph_module._init_node(state) == {"relay_plugins_path": path if exists else None}


def test_init_without_a_plugins_path_yields_none() -> None:
    assert graph_module._init_node(graph_state()) == {"relay_plugins_path": None}
