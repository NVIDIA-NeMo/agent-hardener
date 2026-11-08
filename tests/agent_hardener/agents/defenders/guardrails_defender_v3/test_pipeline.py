# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The whole defender: what a run leaves behind, and how it is chosen over v2."""

from __future__ import annotations

import tomllib
from typing import TYPE_CHECKING

import pytest

from agent_hardener.agents.defenders import guardrails_defender
from agent_hardener.agents.defenders.guardrails_defender_v3 import summary
from agent_hardener.agents.defenders.guardrails_defender_v3.guardrails_defender_agent import run
from agent_hardener.agents.defenders.guardrails_defender_v3.schemas import AnalysisResponse, RailResponse
from agent_hardener.relay_plugin.config import judge_api_key_env

from .conftest import (
    RAIL,
    FakeLLM,
    artifacts,
    bash_trace,
    defender_input,
    found,
    outcome,
    trace,
    written_rails,
)

if TYPE_CHECKING:
    from pathlib import Path

FOUND = found(2, "bash_executor", ["command"])


def test_every_phase_leaves_an_artifact_and_a_readable_summary(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)

    run(inp, llm=FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]}), grader=FakeLLM({}))

    files = sorted(p.name for p in artifacts(tmp_path, inp).iterdir())
    assert files == [
        "00_input.json",
        "01_prepare.json",
        "02_vulnerability_analysis.attempt1.json",
        "03_benign_args.json",
        "04_rail.json",
        "outcome.json",
        "summary.md",
    ]
    text = (artifacts(tmp_path, inp) / "summary.md").read_text()
    assert "**Outcome:** applied" in text
    assert "- Call: step 2, `bash_executor`" in text
    assert "### Iteration 1 (score 10.0/10)" in text


def test_rails_accumulate_in_one_plugin_config(tmp_path: Path) -> None:
    llm = FakeLLM({AnalysisResponse: [FOUND, FOUND], RailResponse: [RAIL, RAIL]})

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)
    run(defender_input(bash_trace(), prompt="another attack", tmp=tmp_path), llm=llm, grader=llm)

    assert [(r.name, r.target_tool, r.threshold) for r in written_rails(tmp_path)] == [
        ("custom_guardrail_1", "bash_executor", 0.7),
        ("custom_guardrail_2", "bash_executor", 0.7),
    ]


def test_a_model_failure_is_reported_not_raised(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RuntimeError("503")]})

    output = run(inp, llm=llm, grader=llm)

    assert not output.ok
    assert outcome(tmp_path, inp)["status"] == "error"
    assert "**Outcome:** error" in (artifacts(tmp_path, inp) / "summary.md").read_text()


def test_the_summary_has_one_row_per_attack(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})
    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)
    run(defender_input(trace("no tools"), prompt="other", tmp=tmp_path), llm=llm, grader=llm)

    assert summary.main([str(tmp_path / "artifacts")]) == 0
    assert "2 attacks: applied=1, skipped=1 | routes: tool=1" in capsys.readouterr().out
    assert summary.main([str(tmp_path / "nothing")]) == 1


def test_an_unknown_version_is_an_error(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    inp.context[guardrails_defender.GUARDRAILS_VERSION_KEY] = "v9"

    with pytest.raises(ValueError, match="unknown guardrails_version 'v9'"):
        guardrails_defender.run(inp)


def test_the_judge_is_the_defenders_safety_llm_with_the_v2_key_and_threshold(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    inp.context["safety_llm"] = "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    assert run(inp, llm=llm, grader=llm).ok

    config = tomllib.loads((tmp_path / "plugins.toml").read_text())["components"][0]["config"]
    assert config["model"]["model"] == "nvidia/nvidia/Nemotron-3-Nano-30B-A3B"
    assert config["model"]["api_key_env"] == judge_api_key_env()
    assert "stage" not in config["guardrails"][0]
