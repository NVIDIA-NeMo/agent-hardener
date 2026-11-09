# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Writing a rail into the run's Relay plugin config: the file the victim loads at startup."""

from __future__ import annotations

import threading
import tomllib
from typing import TYPE_CHECKING

from agent_hardener.agents.defenders.traces_guardrails_defender.nodes.component_writer import append_guardrail
from agent_hardener.relay_plugin.config import PLUGIN_KIND, GuardrailsComponentConfig

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _component(target: Path) -> dict:
    document = tomllib.loads(target.read_text(encoding="utf-8"))  # raises if the TOML is corrupt
    return next(entry for entry in document["components"] if entry["kind"] == PLUGIN_KIND)["config"]


def _append(target: Path, tool: str = "transfer_funds", instructions: str = "Block unrequested transfers.") -> str:
    name, _rendered = append_guardrail(target, tool=tool, instructions=instructions, attack_id="atk")
    return name


def test_writes_a_guardrail_the_plugin_can_load(tmp_path: Path) -> None:
    """The writer and the plugin share one schema; a mismatch here is a victim that will not start."""
    target = tmp_path / "plugins.toml"
    target.write_text("version = 1\n", encoding="utf-8")

    _append(target)

    config = GuardrailsComponentConfig.model_validate(_component(target))
    assert [(rail.name, rail.target_tool) for rail in config.guardrails] == [("custom_guardrail_1", "transfer_funds")]


def test_the_api_key_is_never_written_only_its_env_var_name(tmp_path: Path) -> None:
    """This file is uploaded into the victim image and rendered in run reports."""
    target = tmp_path / "plugins.toml"
    _append(target)
    model = _component(target)["model"]
    assert model["api_key_env"] == "INFERENCE_API_KEY"
    assert "api_key" not in model


def test_the_judge_uses_its_own_endpoint_and_key_when_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A safety endpoint must win over the analysis one, and the judge must read the matching key."""
    monkeypatch.setenv("AGENT_HARDENER_BASE_URL", "https://analysis.example/v1")
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_BASE_URL", "http://10.0.0.5:8000/v1")
    monkeypatch.setenv("AGENT_HARDENER_SAFETY_API_KEY", "safety-key")
    target = tmp_path / "plugins.toml"

    _append(target)

    model = _component(target)["model"]
    assert model["base_url"] == "http://10.0.0.5:8000/v1"
    assert model["api_key_env"] == "AGENT_HARDENER_SAFETY_API_KEY"
    assert "safety-key" not in target.read_text(encoding="utf-8")


def test_the_judge_falls_back_to_the_analysis_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENT_HARDENER_SAFETY_BASE_URL", raising=False)
    monkeypatch.delenv("AGENT_HARDENER_SAFETY_API_KEY", raising=False)
    monkeypatch.setenv("AGENT_HARDENER_BASE_URL", "https://analysis.example/v1")
    target = tmp_path / "plugins.toml"

    _append(target)

    model = _component(target)["model"]
    assert model["base_url"] == "https://analysis.example/v1"
    assert model["api_key_env"] == "INFERENCE_API_KEY"


def test_rails_append_into_the_one_component_entry(tmp_path: Path) -> None:
    """One entry per kind: a second component would register the same guardrail names twice."""
    target = tmp_path / "plugins.toml"
    _append(target, tool="transfer_funds")
    _append(target, tool="read_file")

    document = tomllib.loads(target.read_text(encoding="utf-8"))
    assert len(document["components"]) == 1
    config = GuardrailsComponentConfig.model_validate(_component(target))
    assert [(rail.name, rail.target_tool) for rail in config.guardrails] == [
        ("custom_guardrail_1", "transfer_funds"),
        ("custom_guardrail_2", "read_file"),
    ]


def test_concurrent_writes_stay_valid_and_lose_none(tmp_path: Path) -> None:
    """The manager runs one defender per attack, each in its own thread, all writing this file."""
    target = tmp_path / "plugins.toml"
    target.write_text("version = 1\n", encoding="utf-8")

    threads = [threading.Thread(target=_append, args=(target, f"tool_{index}")) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    config = GuardrailsComponentConfig.model_validate(_component(target))
    assert len(config.guardrails) == 16  # none clobbered
    assert len({rail.name for rail in config.guardrails}) == 16  # and none reused a name


def test_a_backslash_in_the_instruction_survives_verbatim(tmp_path: Path) -> None:
    target = tmp_path / "plugins.toml"
    _append(target, instructions=r"block reads under C:\Users\ and ~/.ssh")
    config = GuardrailsComponentConfig.model_validate(_component(target))
    assert config.guardrails[0].system_instructions == r"block reads under C:\Users\ and ~/.ssh"
