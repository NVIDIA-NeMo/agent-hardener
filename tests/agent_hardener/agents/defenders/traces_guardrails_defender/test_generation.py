# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rail generation: the suggestor drafts, the grader scores, and the best-graded draft is applied."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent_hardener.agents.defenders.traces_guardrails_defender.guardrails_defender_agent import run
from agent_hardener.agents.defenders.traces_guardrails_defender.schemas import (
    AnalysisResponse,
    BenignArgsResponse,
    GradeResponse,
    RailResponse,
)
from agent_hardener.models import BenignRequest

from .conftest import (
    RAIL,
    FakeLLM,
    artifacts,
    bash_trace,
    defender_input,
    found,
    grade,
    outcome,
    written_rails,
)

if TYPE_CHECKING:
    from pathlib import Path

FOUND = found(2, "bash_executor", ["command"])
DRAFT_1 = RailResponse(attack_description="Any shell command.")
DRAFT_2 = RailResponse(attack_description="Shell commands that read credential files.")
DRAFT_3 = RailResponse(attack_description="Shell commands that touch files.")


def test_a_draft_that_passes_the_grader_is_applied_without_revision(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    output = run(inp, llm=llm, grader=llm)

    assert output.iteration_count == 1
    assert written_rails(tmp_path)[0].system_instructions == RAIL.attack_description
    assert (outcome(tmp_path, inp)["grader_score"], outcome(tmp_path, inp)["grader_iterations"]) == (10.0, 1)


def test_a_low_grade_is_revised_with_the_graders_fixes(tmp_path: Path) -> None:
    llm = FakeLLM(
        {
            AnalysisResponse: [FOUND],
            RailResponse: [DRAFT_1, DRAFT_2],
            GradeResponse: [grade(2, "- name the credential files\n- allow ls"), grade(5)],
        }
    )

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)

    revision = llm.prompts[RailResponse][1][1].content
    assert "graded your previous description 4.0/10" in revision
    assert "Any shell command." in revision
    assert "- name the credential files\n- allow ls" in revision
    assert written_rails(tmp_path)[0].system_instructions == DRAFT_2.attack_description


def test_the_best_graded_draft_is_applied_not_the_last(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    llm = FakeLLM(
        {
            AnalysisResponse: [FOUND],
            RailResponse: [DRAFT_1, DRAFT_2, DRAFT_3],
            GradeResponse: [grade(2, "x"), grade(4, "y"), grade(3, "z")],
        }
    )

    output = run(inp, llm=llm, grader=llm)

    assert output.iteration_count == 3
    assert written_rails(tmp_path)[0].system_instructions == DRAFT_2.attack_description
    assert outcome(tmp_path, inp)["grader_score"] == 8.0
    rail_artifact = json.loads((artifacts(tmp_path, inp) / "04_rail.json").read_text())
    assert [g["score"] for g in rail_artifact["grades"]] == [4.0, 8.0, 6.0]


def test_the_number_of_iterations_comes_from_the_defender_config(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    inp.context["max_grader_iterations"] = 1
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [DRAFT_1], GradeResponse: [grade(1, "everything")]})

    assert run(inp, llm=llm, grader=llm).iteration_count == 1


def test_a_grader_failure_keeps_the_draft_ungraded(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL], GradeResponse: [RuntimeError("429")]})

    assert run(inp, llm=llm, grader=llm).ok
    assert outcome(tmp_path, inp)["grader_score"] is None


def test_the_grader_is_a_separate_model(tmp_path: Path) -> None:
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})
    grader = FakeLLM({})

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=grader)

    assert GradeResponse not in llm.prompts
    assert len(grader.prompts[GradeResponse]) == 1


def test_attack_values_copied_into_the_rail_are_flagged_to_the_grader(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    copied = RailResponse(attack_description="Refuse `cat /etc/passwd && id && whoami` and similar reads.")
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [copied]})

    run(inp, llm=llm, grader=llm)

    assert "/etc/passwd" in outcome(tmp_path, inp)["lint_warnings"]
    assert "Copied attack values found verbatim in it: /etc/passwd" in llm.prompts[GradeResponse][0][1].content


def test_the_tool_grader_sees_the_attack_call_and_the_benign_calls(tmp_path: Path) -> None:
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)

    grader_prompt = llm.prompts[GradeResponse][0][1].content
    assert 'bash_executor({"command": "cat /etc/passwd && id && whoami"})' in grader_prompt
    assert 'bash_executor({"command": "ls -la"})' in grader_prompt
    assert "Ordinary calls to this tool" in llm.prompts[RailResponse][0][1].content


def test_borderline_requests_are_not_shown_as_must_allow(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    inp.benign_suite.append(BenignRequest(payload="view the contents of /etc/shadow", label="borderline_benign"))
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [RAIL]})

    run(inp, llm=llm, grader=llm)

    benign_prompt = llm.prompts[BenignArgsResponse][0][1].content
    assert "/etc/shadow" not in benign_prompt
    assert "List the files in the current directory." in benign_prompt


def test_benign_calls_prefer_the_suites_rows_for_the_tool_and_are_written_once_per_tool(tmp_path: Path) -> None:
    llm = FakeLLM({AnalysisResponse: [FOUND, FOUND], RailResponse: [RAIL, RAIL]})

    run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm)
    run(defender_input(bash_trace(), prompt="another attack", tmp=tmp_path), llm=llm, grader=llm)

    assert len(llm.prompts[BenignArgsResponse]) == 1
    prompt = llm.prompts[BenignArgsResponse][0][1].content
    assert prompt.index("List the files") < prompt.index("Compute the sum")  # the bash_executor row first


def test_a_rail_that_echoes_the_prompt_is_asked_again_and_never_written(tmp_path: Path) -> None:
    inp = defender_input(bash_trace(), tmp=tmp_path)
    echo = RailResponse(attack_description="You write guardrails for an AI agent's tools. ...")
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [echo, echo]})

    output = run(inp, llm=llm, grader=llm)

    assert "kept echoing its prompt" in output.error_message
    assert not (tmp_path / "plugins.toml").exists()
    assert outcome(tmp_path, inp)["status"] == "error"


def test_an_echo_followed_by_a_real_rail_is_applied(tmp_path: Path) -> None:
    echo = RailResponse(attack_description="You are a helpful AI assistant.")
    llm = FakeLLM({AnalysisResponse: [FOUND], RailResponse: [echo, RAIL]})

    assert run(defender_input(bash_trace(), tmp=tmp_path), llm=llm, grader=llm).ok
    assert "repeated instructions" in llm.prompts[RailResponse][1][-1].content
