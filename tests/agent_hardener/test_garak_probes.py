# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The garak probe registry: one entry names a probe everywhere the attacker and the replay use it."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from agent_hardener.agents.attackers.agent_breaker import preflight
from agent_hardener.agents.attackers.agent_breaker.config import apply_runtime_fields, build_agent_breaker_config
from agent_hardener.agents.attackers.agent_breaker.diagnosis import ScanEvidence, diagnose_scan
from agent_hardener.agents.attackers.garak_probes import (
    AGENT_BREAKER,
    DETECTORS,
    PROBES,
    GarakProbe,
    split_plugin,
)
from agent_hardener.agents.validators.garak_replay import parsing

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
VICTIM = "http://victim:8000/v1/chat/completions"

#: A second probe, shaped like a real registry entry, to show nothing is tied to AgentBreaker's names.
OTHER = GarakProbe(
    key="tool_chain",
    probe="tool_chain.ToolChain",
    detector="tool_chain.ToolChainResult",
    report_prefix="tool-chain",
    attack_type="direct_prompt_injection",
)


def test_agent_breaker_config_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [*os.environ]:
        if name.startswith("GARAK_"):
            monkeypatch.delenv(name)
    expected = yaml.safe_load((FIXTURES / "agent_breaker.resolved.yaml").read_text(encoding="utf-8"))

    built = build_agent_breaker_config(target_uri=VICTIM, report_dir="/runs/garak", report_prefix="agent-breaker")

    assert built == expected


def test_agent_breaker_keeps_its_file_names_and_source() -> None:
    assert (
        AGENT_BREAKER.source,
        AGENT_BREAKER.stdout_log_name,
        AGENT_BREAKER.detail_log_name,
        AGENT_BREAKER.resolved_config_name,
        AGENT_BREAKER.probe_class,
    ) == (
        "garak-agent-breaker",
        "garak-agent-breaker.log",
        "garak-agent-breaker.detail.log",
        "garak-agent-breaker.resolved.yaml",
        "AgentBreaker",
    )


def test_every_registered_probe_has_a_replay_detector() -> None:
    assert {probe.attack_type for probe in PROBES.values()} <= set(DETECTORS)
    assert all(DETECTORS[probe.attack_type] == probe.detector for probe in PROBES.values())


@pytest.mark.parametrize("name", ["AgentBreaker", "agent_breaker.", ".AgentBreaker", ""])
def test_a_plugin_name_without_module_and_class_is_rejected(name: str) -> None:
    with pytest.raises(ValueError, match=r"<module>\.<Class>"):
        split_plugin(name)


def test_another_probe_gets_its_own_plugin_blocks_and_model_timeouts() -> None:
    built = build_agent_breaker_config(
        target_uri=VICTIM, report_dir="d", report_prefix="tool-chain", overrides={"model_timeout": 900}, probe=OTHER
    )
    plugins = built["plugins"]

    assert (plugins["probe_spec"], plugins["extended_detectors"]) == (
        "tool_chain.ToolChain",
        ["tool_chain.ToolChainResult"],
    )
    assert set(plugins["probes"]) == {"tool_chain"}
    assert set(plugins["detectors"]) == {"tool_chain"}

    resolved = apply_runtime_fields(
        built, target_uri=VICTIM, report_dir="d", report_prefix="p", model_timeout=1200, probe=OTHER
    )
    probe_block = OTHER.plugin_config(resolved["plugins"], "probes")
    detector_block = OTHER.plugin_config(resolved["plugins"], "detectors")
    assert probe_block["red_team_model_config"]["timeout"] == detector_block["detector_model_config"]["timeout"] == 1200


def test_preflight_checks_the_models_of_the_probe_that_runs() -> None:
    config = build_agent_breaker_config(
        target_uri=VICTIM, report_dir="d", report_prefix="p", overrides={"red_team_model_name": "a/b"}, probe=OTHER
    )

    assert {call.label for call in preflight._model_calls(config, OTHER)} == {"red-team", "detector"}
    assert preflight._model_calls(config, AGENT_BREAKER) == []


def test_diagnosis_reads_the_log_lines_of_the_probe_that_ran() -> None:
    log = "\n".join(
        [
            "2026-11-04 18:25:22,675  INFO  invoked",
            "2026-11-18 22:25:49,573  INFO  ToolChain # Attacking 3 tools",
        ]
    )

    failure = diagnose_scan(ScanEvidence(), log, target_uri=VICTIM, probe=OTHER)

    assert failure is not None
    assert failure.code == "no_attacks_generated"
    assert "3 tool(s)" in failure.message


def test_replay_names_the_detector_from_the_registry() -> None:
    assert parsing.detector_namespace_and_class("direct_prompt_injection") == ("agent_breaker", "AgentBreakerResult")
    assert parsing.detector_namespace_and_class("indirect_prompt_injection") == (
        "indirect_injection",
        "IndirectInjectionResult",
    )
