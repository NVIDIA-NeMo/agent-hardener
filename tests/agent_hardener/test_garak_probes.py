# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The garak probe registry: one entry names a probe everywhere the attacker and the replay use it."""

from __future__ import annotations

import json
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
    INDIRECT_PROBE,
    PROBES,
    GarakProbe,
    split_plugin,
)
from agent_hardener.agents.attackers.preloaded import load_preloaded_attacks
from agent_hardener.agents.validators.garak_replay import parsing
from agent_hardener.final_log import SCANNER_LABELS
from agent_hardener.final_log.helpers import _classify_scanner
from agent_hardener.models import PreloadedAttackConfig

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


def test_the_profile_file_is_set_on_the_probe_only_when_given() -> None:
    built = build_agent_breaker_config(target_uri=VICTIM, report_dir="d", report_prefix="p", probe=OTHER)
    plain = apply_runtime_fields(built, target_uri=VICTIM, report_dir="d", report_prefix="p", probe=OTHER)
    assert "agent_config_file" not in OTHER.plugin_config(plain["plugins"], "probes")

    pointed = apply_runtime_fields(
        built, target_uri=VICTIM, report_dir="d", report_prefix="p", probe=OTHER, agent_config_file="ah/x/agent.yaml"
    )
    assert OTHER.plugin_config(pointed["plugins"], "probes")["agent_config_file"] == "ah/x/agent.yaml"
    assert "agent_config_file" not in AGENT_BREAKER.plugin_config(pointed["plugins"], "probes")


def test_scanner_labels_follow_the_registry() -> None:
    assert list(SCANNER_LABELS) == ["indirect", "agent_breaker", "other"]


@pytest.mark.parametrize(
    ("values", "scanner"),
    [
        (["garak-agent-breaker.report.jsonl"], "agent_breaker"),
        (["agent_breaker.AgentBreaker"], "agent_breaker"),
        (["AgentBreakerResult"], "agent_breaker"),
        (["latentinjection.IndirectInjection"], "indirect"),
        (["indirect_injection.IndirectInjectionResult"], "indirect"),
        (["prompt_injection"], "indirect"),
        (["encoding.InjectBase64"], "other"),
    ],
)
def test_scanner_classification_is_unchanged(values: list[str], scanner: str) -> None:
    assert _classify_scanner(values) == scanner


def test_a_registered_probe_gets_its_own_scanner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(PROBES, OTHER.key, OTHER)
    assert _classify_scanner(["garak-tool-chain.hitlog.jsonl"]) == "tool_chain"
    assert _classify_scanner(["tool_chain.ToolChain"]) == "tool_chain"
    assert _classify_scanner(["ToolChainResult"]) == "tool_chain"


def test_a_hits_garak_probe_decides_its_attack_type_over_its_source() -> None:
    indirect = {"probe": INDIRECT_PROBE, "detector": "", "source": AGENT_BREAKER.source}
    assert parsing.identify_attack_type(indirect) == "indirect_prompt_injection"
    direct = {"probe": AGENT_BREAKER.probe, "source": "garak-indirect-injection"}
    assert parsing.identify_attack_type(direct) == "direct_prompt_injection"
    assert parsing.identify_attack_type({"source": AGENT_BREAKER.source}) == "direct_prompt_injection"
    assert parsing.identify_attack_type({"probe": "encoding.InjectBase64"}) is None


def test_a_replayed_hit_takes_its_probes_source(tmp_path: Path) -> None:
    hitlog = tmp_path / "mixed.hitlog.jsonl"
    rows = [
        {"probe": AGENT_BREAKER.probe, "prompt": "a"},
        {"probe": INDIRECT_PROBE, "prompt": "b"},
        {"probe": INDIRECT_PROBE, "prompt": "c", "source": "garak-indirect-injection"},
    ]
    hitlog.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    inferred = load_preloaded_attacks([PreloadedAttackConfig(name="replay", path=hitlog)])
    assert [attack.records[0].get("source") for attack in inferred] == [
        AGENT_BREAKER.source,
        None,
        "garak-indirect-injection",
    ]
    assert [parsing.identify_attack_type(attack.records[0]) for attack in inferred] == [
        "direct_prompt_injection",
        "indirect_prompt_injection",
        "indirect_prompt_injection",
    ]

    stamped = load_preloaded_attacks([PreloadedAttackConfig(name="replay", path=hitlog, source="custom")])
    assert [attack.records[0]["source"] for attack in stamped] == ["custom", "custom", "garak-indirect-injection"]


def test_replay_names_the_detector_from_the_registry() -> None:
    assert parsing.detector_namespace_and_class("direct_prompt_injection") == ("agent_breaker", "AgentBreakerResult")
    assert parsing.detector_namespace_and_class("indirect_prompt_injection") == (
        "indirect_injection",
        "IndirectInjectionResult",
    )
