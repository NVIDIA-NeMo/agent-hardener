# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.defenders.guardrails_defender_v2 import guardrails_defender_agent as agent
from agent_hardener.agents.defenders.guardrails_defender_v2.nodes import (
    custom_guardrails_generator as gen,
)
from agent_hardener.agents.defenders.guardrails_defender_v2.nodes import (
    vulnerability_analyzer as va,
)
from agent_hardener.models import DefenderOutput
from agent_hardener.relay_plugin.config import DEFAULT_GUARDRAIL_THRESHOLD

from .fakes import defender_input

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from .conftest import InstallLLM


class _FakeGraph:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.initial_state: Any = None

    def invoke(self, state: Any) -> dict[str, Any]:
        self.initial_state = state
        return self.result


def _use_graph(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> _FakeGraph:
    fake = _FakeGraph(result)
    monkeypatch.setattr(agent, "build_defender_graph", lambda: fake)
    return fake


def test_run_seeds_the_graph_from_the_defender_input_and_returns_its_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    expected = DefenderOutput(ok=True, new_policy_yaml="x = 1")
    fake = _use_graph(monkeypatch, {"final_output": expected})
    plugins = tmp_path / "plugins.toml"
    defender = defender_input(
        relay_plugins_path=plugins, context={"current_config": "version = 1", "safety_llm": "judge-model"}
    )

    assert agent.run(defender) is expected

    state = fake.initial_state
    assert state.original_input is defender
    assert state.current_config == "version = 1"
    assert state.safety_llm == "judge-model"
    assert state.relay_plugins_path == plugins


def test_run_defaults_optional_context_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _use_graph(monkeypatch, {"final_output": DefenderOutput(ok=True)})

    agent.run(defender_input())

    assert fake.initial_state.current_config == ""
    assert fake.initial_state.safety_llm is None
    assert fake.initial_state.relay_plugins_path is None


def test_run_reports_a_graph_that_produced_no_output(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_graph(monkeypatch, {"final_output": None})

    result = agent.run(defender_input())

    assert result.ok is False
    assert result.error_message == "Graph completed without producing output."


def test_run_passes_through_a_failed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    failed = DefenderOutput(ok=False, error_message="nothing to guard")
    _use_graph(monkeypatch, {"final_output": failed})

    assert agent.run(defender_input()) is failed


def test_end_to_end_writes_the_generated_guardrail_into_the_plugins_file(stub_llm: InstallLLM, tmp_path: Path) -> None:
    stub_llm(va, lambda _text: "analysis")
    stub_llm(gen, lambda text: "[SCORE]: 10" if gen.GRADER_SYS_PROMPT in text else "Score 1.0 for credential reads.")
    plugins = tmp_path / "plugins.toml"
    plugins.write_text("version = 1\n", encoding="utf-8")
    defender = defender_input(relay_plugins_path=plugins, context={"safety_llm": "judge-model", "attack_id": "atk-7"})

    result = agent.run(defender)

    assert result.ok is True
    assert result.guardrail_name == "custom_guardrail_1"
    config = tomllib.loads(plugins.read_text(encoding="utf-8"))["components"][0]["config"]
    assert config["model"]["model"] == "judge-model"
    assert config["guardrails"] == [
        {
            "name": "custom_guardrail_1",
            "target_tool": "bash",
            "action": "refusal",
            "threshold": DEFAULT_GUARDRAIL_THRESHOLD,
            "system_instructions": "Score 1.0 for credential reads.",
            "attack_id": "atk-7",
        }
    ]
