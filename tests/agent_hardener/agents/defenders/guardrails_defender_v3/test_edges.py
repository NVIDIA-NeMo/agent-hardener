# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edge cases: odd model answers, missing benign calls, and recording that must never fail the defender."""

from __future__ import annotations

import json
import runpy
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.agents.defenders.guardrails_defender_v3 import llm as llm_module
from agent_hardener.agents.defenders.guardrails_defender_v3.guardrails_defender_agent import run
from agent_hardener.agents.defenders.guardrails_defender_v3.nodes.benign_args import benign_args_node
from agent_hardener.agents.defenders.guardrails_defender_v3.nodes.component_writer import component_writer_node
from agent_hardener.agents.defenders.guardrails_defender_v3.nodes.guardrail_generator import (
    copied_literal_candidates,
    guardrail_generator_node,
)
from agent_hardener.agents.defenders.guardrails_defender_v3.nodes.prepare import prepare_node
from agent_hardener.agents.defenders.guardrails_defender_v3.observability import (
    Recorder,
    artifacts_dir,
)
from agent_hardener.agents.defenders.guardrails_defender_v3.prompting import as_text
from agent_hardener.agents.defenders.guardrails_defender_v3.schemas import (
    AnalysisResponse,
    BenignArgsResponse,
    BenignCallResponse,
    GradeResponse,
    RailResponse,
)
from agent_hardener.agents.defenders.guardrails_defender_v3.state import DefenderState

from .conftest import (
    BENIGN,
    RAIL,
    FakeLLM,
    artifacts,
    bash_trace,
    defender_input,
    found,
    grade,
    outcome,
    tool,
    trace,
)

if TYPE_CHECKING:
    from pathlib import Path

FOUND = found(2, "bash_executor", ["command"])


def test_a_tool_name_that_does_not_match_the_step_is_retried(tmp_path: Path) -> None:
    llm = FakeLLM({AnalysisResponse: [found(2, "python_executor", ["command"]), FOUND], RailResponse: [RAIL]})

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)

    assert "step 2 called bash_executor, not python_executor" in llm.prompts[AnalysisResponse][1][1].content


def test_answers_returned_as_plain_dicts_are_accepted(tmp_path: Path) -> None:
    """Some providers hand structured output back as a dict rather than the schema's model."""
    llm = FakeLLM(
        {
            AnalysisResponse: [FOUND.model_dump()],
            BenignArgsResponse: [BenignArgsResponse(calls=[]).model_dump()],
            RailResponse: [RAIL.model_dump()],
            GradeResponse: [grade(5).model_dump()],
        }
    )

    assert run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm).ok


def test_benign_calls_with_unusable_arguments_are_dropped(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    calls = BenignArgsResponse(
        calls=[
            BenignCallResponse(request=BENIGN[1], args_json='{"command": "ls"}'),
            BenignCallResponse(request=BENIGN[0], args_json='{"cmd": "ls"}'),
            BenignCallResponse(request=BENIGN[0], args_json="not json"),
        ]
    )
    llm = FakeLLM({AnalysisResponse: [FOUND], BenignArgsResponse: [calls], RailResponse: [RAIL]})

    run(inp, llm=llm, grader=llm)

    assert [c.args for c in _benign_calls(tmp_path, inp)] == [{"command": "ls"}]


def test_without_benign_calls_the_suggestor_sees_the_benign_requests(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    llm = FakeLLM({AnalysisResponse: [FOUND], BenignArgsResponse: [RuntimeError("503")], RailResponse: [RAIL]})

    assert run(inp, llm=llm, grader=llm).ok
    assert "Examples of ordinary requests users make" in llm.prompts[RailResponse][0][1].content


def test_an_empty_benign_suite_writes_no_benign_calls(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    inp.benign_suite, inp.benign_requests = [], []
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    assert run(inp, llm=llm, grader=llm).ok
    assert BenignArgsResponse not in llm.prompts


def test_list_argument_values_are_linted_too(tmp_path: Path) -> None:
    attack = trace("x", tool("bash_executor", {"argv": ["cat", "/etc/shadow-backup"]}))
    copied = RailResponse(attack_description="Reads of /etc/shadow-backup.")
    llm = FakeLLM({AnalysisResponse: [found(1, "bash_executor", ["argv"])], RailResponse: [copied]})
    inp = defender_input(attack, tmp=tmp_path)

    run(inp, llm=llm, grader=llm)

    assert outcome(tmp_path, inp)["lint_warnings"] == ["/etc/shadow-backup"]


def test_the_default_model_serves_every_role_and_grades_at_temperature_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[tuple[str | None, float]] = []
    fake = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    def build(*, temperature: float, model: str | None = None, **_: Any) -> FakeLLM:
        built.append((model, temperature))
        return fake

    monkeypatch.setattr(llm_module, "build_chat_model", build)

    assert run(defender_input(bash_trace(), tmp=tmp_path)).ok
    assert built == [(None, 0.2), (None, 0.0)]


def test_artifacts_go_next_to_the_runs_plugin_config(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    del inp.context["defender_v3_artifacts_dir"]
    inp.relay_plugins_path = tmp_path / "run" / "victim-active-state" / "plugins.toml"

    assert artifacts_dir(inp, "a") == tmp_path / "run" / "defender_v3" / "round_1" / "a"
    inp.relay_plugins_path = None
    assert artifacts_dir(inp, "a") is None


def test_recording_never_fails_the_defender(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("")

    Recorder(blocker / "sub").write("input", {"a": 1})  # mkdir under a file: logged, not raised
    Recorder(None).write("input", {"a": 1})

    with pytest.raises(ValueError, match="unknown defender v3 phase"):
        Recorder(None).write("nope", {})


def test_arguments_that_do_not_serialise_are_still_shown() -> None:
    loop: list[Any] = []
    loop.append(loop)
    assert as_text(loop) == "[[...]]"


def _benign_calls(tmp_path: Path, inp: Any) -> list[Any]:
    data = json.loads((artifacts(tmp_path, inp) / "03_benign_args.json").read_text())
    return [SimpleNamespace(**call) for call in data["benign_calls"]]


def test_a_node_reached_without_a_tool_analysis_fails_loudly(tmp_path: Path) -> None:
    """Routing never sends such a state there; if it ever did, the defender must not write a rail."""
    state = DefenderState(input=defender_input(bash_trace(), tmp=tmp_path), attack="a")
    llm = FakeLLM({})
    with pytest.raises(ValueError, match="needs a tool analysis"):
        benign_args_node(state, llm=llm)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs a tool analysis"):
        guardrail_generator_node(state, llm=llm, grader=llm)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="needs a rail"):
        component_writer_node(state)
    untraced = DefenderState(input=defender_input(None, tmp=tmp_path), attack="a")
    with pytest.raises(ValueError, match="needs the attack's trace"):
        prepare_node(untraced)


def test_only_string_argument_values_are_literals() -> None:
    assert copied_literal_candidates({"path": "/etc/shadow", "count": 5, "flags": ["-la"]}) == {"/etc/shadow", "-la"}


def test_the_summary_runs_as_a_module(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_:
        runpy.run_module(
            "agent_hardener.agents.defenders.guardrails_defender_v3.summary", run_name="__main__", alter_sys=True
        )
    assert exit_.value.code == 2  # argparse: the root argument is required
