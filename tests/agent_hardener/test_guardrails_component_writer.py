# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Tests for writing a generated guardrail into the run's Relay plugin config.

The NAT-era writer patched the victim's own workflow YAML and needed two tests this file drops: the
``safety_llm`` injection (the judge model is the component's own config now, so nothing is written
into the agent's configuration at all) and the backslash normalisation (NAT's FastAPI front-end
re-serialised the config as a double-quoted YAML scalar, where a lone ``\\U`` from a Windows path was
an invalid unicode escape and stopped the victim serving; TOML has no such quirk).
"""

from __future__ import annotations

import threading
import tomllib
from typing import TYPE_CHECKING

from agent_hardener.agents.defenders.guardrails_defender_v2.nodes.component_writer import (
    COMPONENT_KIND,
    component_writer_node,
)
from agent_hardener.agents.defenders.guardrails_defender_v2.schemas import GraphState
from agent_hardener.models import DefenderInput
from agent_hardener.relay_plugin.config import GuardrailsComponentConfig

if TYPE_CHECKING:
    from pathlib import Path


def _state(
    target: Path, *, tool: str = "transfer_funds", instruction: str = "Block unrequested transfers."
) -> GraphState:
    return GraphState(
        original_input=DefenderInput(attack_prompt="a", agent_response="b", attacked_tool=tool),
        relay_plugins_path=target,
        draft_instruction=instruction,
    )


def _component(document: dict) -> dict:
    entry = next(e for e in document["components"] if e["kind"] == COMPONENT_KIND)
    return entry["config"]


def test_writes_a_guardrail_the_plugin_can_load(tmp_path: Path) -> None:
    """The writer and the plugin share one schema; a mismatch here is a victim that will not start."""
    target = tmp_path / "plugins.toml"
    target.write_text("version = 1\n", encoding="utf-8")

    component_writer_node(_state(target))

    document = tomllib.loads(target.read_text(encoding="utf-8"))
    config = GuardrailsComponentConfig.model_validate(_component(document))
    assert [rail.name for rail in config.guardrails] == ["custom_guardrail_1"]
    assert config.guardrails[0].target_tool == "transfer_funds"


def test_the_api_key_is_never_written_only_its_env_var_name(tmp_path: Path) -> None:
    """This file is uploaded into the victim image and rendered in run reports."""
    target = tmp_path / "plugins.toml"
    component_writer_node(_state(target))
    model = _component(tomllib.loads(target.read_text(encoding="utf-8")))["model"]
    assert model["api_key_env"] == "INFERENCE_API_KEY"
    assert "api_key" not in model


def test_successive_rounds_append_rather_than_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "plugins.toml"
    component_writer_node(_state(target, tool="transfer_funds"))
    component_writer_node(_state(target, tool="read_file"))

    config = GuardrailsComponentConfig.model_validate(_component(tomllib.loads(target.read_text(encoding="utf-8"))))
    assert [rail.name for rail in config.guardrails] == ["custom_guardrail_1", "custom_guardrail_2"]
    assert [rail.target_tool for rail in config.guardrails] == ["transfer_funds", "read_file"]


def test_a_second_round_reuses_the_one_component_entry(tmp_path: Path) -> None:
    """One entry per kind: a second component would re-register the same guardrail names twice."""
    target = tmp_path / "plugins.toml"
    component_writer_node(_state(target))
    component_writer_node(_state(target))
    assert len(tomllib.loads(target.read_text(encoding="utf-8"))["components"]) == 1


def test_concurrent_writes_stay_valid_and_lose_none(tmp_path: Path) -> None:
    """The manager fans out one graph per attack, each in its own thread, all writing this file.

    The lock must serialize read→append→save: without it, injections interleave and get clobbered.
    """
    target = tmp_path / "plugins.toml"
    target.write_text("version = 1\n", encoding="utf-8")

    threads = [
        threading.Thread(target=lambda i=index: component_writer_node(_state(target, tool=f"tool_{i}")))
        for index in range(16)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    document = tomllib.loads(target.read_text(encoding="utf-8"))  # raises if the TOML corrupted
    config = GuardrailsComponentConfig.model_validate(_component(document))
    assert len(config.guardrails) == 16  # every concurrent write accumulated; none clobbered
    assert len({rail.name for rail in config.guardrails}) == 16  # and none reused a name


def test_an_unattributable_attack_writes_no_guardrail(tmp_path: Path) -> None:
    """A guardrail with no target guards nothing but would still read as a mitigation in the report."""
    target = tmp_path / "plugins.toml"
    result = component_writer_node(_state(target, tool=""))["final_output"]
    assert result.ok is False
    assert "nothing to guard" in result.error_message
    assert not target.exists()  # nothing written at all


def test_a_backslash_in_the_instruction_survives_verbatim(tmp_path: Path) -> None:
    """NAT had to rewrite these to forward slashes or the victim failed to serve. TOML does not."""
    target = tmp_path / "plugins.toml"
    component_writer_node(_state(target, instruction=r"block reads under C:\Users\ and ~/.ssh"))
    config = GuardrailsComponentConfig.model_validate(_component(tomllib.loads(target.read_text(encoding="utf-8"))))
    assert config.guardrails[0].system_instructions == r"block reads under C:\Users\ and ~/.ssh"
