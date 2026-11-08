# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Phase 2: write the rail for the point the analysis chose, with a suggestor and a grader.

The pattern is v2's (:mod:`agent_hardener.agents.defenders.guardrails_defender_v2.nodes.custom_guardrails_generator`):
a suggestor drafts, a grader scores the draft against a rubric and lists what to fix, and the suggestor
revises, until the score clears :data:`PASS_SCORE` or :data:`MAX_ITERATIONS` drafts were written. v3
grounds both roles in the vulnerability analysis and the trace instead of a free-text report, and
keeps the *best-graded* draft rather than the last one, so a revision that got worse is not applied.

The rail is the ``system_instructions`` the in-victim guard model reads as an "attack description"
before each call to the tool (:func:`agent_hardener.relay_plugin.judge.judge_input`). It is written in
that framing.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel  # noqa: TC002 - LangGraph reads node annotations at runtime
from langchain_core.messages import BaseMessage, HumanMessage

from ..prompting import as_text, messages
from ..schemas import CRITERIA, GradeResponse, RailDraft, RailGrade, RailResponse, VulnerabilityAnalysis
from ..state import DefenderState  # noqa: TC001 - LangGraph reads node annotations at runtime
from .benign_args import benign_requests

logger = logging.getLogger("agent_hardener.session")

#: Suggestor-grader rounds per attack, as in v2.
MAX_ITERATIONS = 3
#: A draft scoring this (out of 10) or more is accepted without another revision, as in v2.
PASS_SCORE = 9.0
#: Benign examples shown to the suggestor and the grader as a picture of normal use.
BENIGN_EXAMPLES = 5
#: Shorter literals are too likely to be ordinary words to flag.
MIN_LITERAL = 8
#: Extra suggestor calls when the answer is the prompt echoed back.
MAX_ECHO_RETRIES = 1
#: A system-prompt line at least this long found in the answer means the answer copied the prompt.
ECHO_LINE = 40


def guardrail_generator_node(state: DefenderState, *, llm: BaseChatModel, grader: BaseChatModel) -> dict[str, Any]:
    """Write a pre-tool rail for the call the analysis grounded."""
    analysis = _analysis(state)
    examples = [f"{call.tool}({as_text(call.args)})" for call in state.benign_calls[:BENIGN_EXAMPLES]]
    context = {
        "tool_name": analysis.tool_name,
        "attack_call": as_text(analysis.call_args),
        "malicious_args": as_text(analysis.malicious_args),
        "attack_pattern": analysis.attack_pattern,
        "benign_contrast": analysis.benign_contrast,
        "attack_prompt": state.input.attack_prompt,
        "trace": state.rendered_trace,
        "benign_examples": examples or benign_requests(state.input)[:BENIGN_EXAMPLES],
        "benign_examples_are_calls": bool(examples),
    }
    literals = copied_literal_candidates(analysis.malicious_args)
    return _suggest_and_grade(state, analysis.tool_name or "", context, literals, llm=llm, grader=grader)


def _analysis(state: DefenderState) -> VulnerabilityAnalysis:
    analysis = state.vulnerability_analysis
    if analysis is None or analysis.target != "tool":  # unreachable: routing sends only tool analyses here
        msg = "the tool rail generator needs a tool analysis"
        raise ValueError(msg)
    return analysis


def _suggest_and_grade(
    state: DefenderState,
    tool: str,
    context: dict[str, Any],
    literals: set[str],
    *,
    llm: BaseChatModel,
    grader: BaseChatModel,
) -> dict[str, Any]:
    """Draft, grade and revise; return every draft and grade, and the best-graded draft as ``rail``."""
    max_iterations = int(state.input.context.get("max_grader_iterations", MAX_ITERATIONS))
    drafts: list[RailDraft] = []
    grades: list[RailGrade] = []
    revision: dict[str, Any] | None = None
    for iteration in range(1, max_iterations + 1):
        prompt = messages("suggest_tool", {**context, "revision": revision})
        response, echo_retries = _ask_suggestor(llm, prompt, tool)
        instructions = response.attack_description.strip()
        draft = RailDraft(
            target_tool=tool,
            system_instructions=instructions,
            rationale=response.rationale,
            iteration=iteration,
            lint_warnings=copied_literals(instructions, literals),
            echo_retries=echo_retries,
        )
        drafts.append(draft)
        grade = _grade(grader, context, draft)
        if grade is None:
            break  # nothing to revise against; keep what is graded so far, or this draft
        grades.append(grade)
        if grade.score >= PASS_SCORE:
            break
        revision = {
            "previous": draft.system_instructions,
            "score": grade.score,
            "fixes": grade.required_fixes,
            "lint_warnings": draft.lint_warnings,
        }
    return {"drafts": drafts, "grades": grades, "rail": best_draft(drafts, grades)}


def best_draft(drafts: list[RailDraft], grades: list[RailGrade]) -> RailDraft:
    """The highest-graded draft; the latest on a tie.

    The last draft when none was graded.
    """
    scores = {grade.iteration: grade.score for grade in grades}
    graded = [draft for draft in drafts if draft.iteration in scores]
    if not graded:
        return drafts[-1]
    return max(graded, key=lambda draft: (scores[draft.iteration], draft.iteration))


def _ask_suggestor(llm: BaseChatModel, prompt: list[BaseMessage], tool: str) -> tuple[RailResponse, int]:
    """The suggestor's answer, asked again (up to :data:`MAX_ECHO_RETRIES` times) while it echoes its prompt.

    Seen from small models on most attacks: the "description" is the system prompt copied back, or a
    generic assistant prompt. Written as a rail, that guards nothing and reads as a mitigation.
    """
    system = str(prompt[0].content)
    subject = f"`{tool}`: which argument values make a call the attack"
    for retries in range(MAX_ECHO_RETRIES + 1):
        response = llm.with_structured_output(RailResponse).invoke(prompt)
        if not isinstance(response, RailResponse):
            response = RailResponse.model_validate(response)
        if not echoes_prompt(response.attack_description, system):
            return response, retries
        prompt = [
            *prompt,
            HumanMessage(
                content="Your answer repeated instructions instead of writing the guardrail. Reply with only the "
                f"attack description for {subject}, and which ordinary ones are not the attack."
            ),
        ]
    msg = f"the suggestor kept echoing its prompt instead of writing a rail ({MAX_ECHO_RETRIES + 1} answers)"
    raise ValueError(msg)


def _grade(grader: BaseChatModel, context: dict[str, Any], draft: RailDraft) -> RailGrade | None:
    """The grader's verdict on ``draft``, or ``None`` when it could not give one.

    A grader failure does not fail the defender: the draft is still the suggestor's best effort, and a
    missing grade shows in the artifacts and the outcome as ``grader_score: null``.
    """
    shown = ("tool_name", "attack_call", "malicious_args", "attack_prompt", "attack_pattern", "benign_contrast")
    prompt = messages(
        "grade",
        {
            **{key: context[key] for key in shown},
            "benign_examples": context["benign_examples"],
            "draft": draft.system_instructions,
            "lint_warnings": draft.lint_warnings,
        },
    )
    try:
        response = grader.with_structured_output(GradeResponse, method="function_calling").invoke(prompt)
        if not isinstance(response, GradeResponse):
            response = GradeResponse.model_validate(response)
    except Exception as exc:
        logger.warning("defender v3 grader failed on iteration %d: %s", draft.iteration, exc)
        return None
    criteria = {name: getattr(response, name) for name in CRITERIA}
    score = round(2 * sum(criteria.values()) / len(criteria), 1)
    fixes = [line.strip(" -*\t") for line in response.required_fixes.splitlines() if line.strip(" -*\t")]
    return RailGrade(iteration=draft.iteration, criteria=criteria, score=score, required_fixes=fixes)


def echoes_prompt(description: str, system_prompt: str) -> bool:
    """Whether ``description`` is the prompt (or an assistant prompt) rather than an attack description."""
    head = description.strip()[:40].lower()
    if head.startswith(("you are", "you write")):
        return True
    lines = (line.strip() for line in system_prompt.splitlines())
    return any(len(line) >= ECHO_LINE and line in description for line in lines)


def copied_literal_candidates(malicious_args: dict[str, Any]) -> set[str]:
    """The attack's argument values, whole and word by word: what a rail should not quote."""
    literals: set[str] = set()
    for value in _strings(malicious_args):
        literals.add(value)
        literals.update(value.split())
    return literals


def copied_literals(instructions: str, literals: set[str]) -> list[str]:
    """Attack literals that appear verbatim in the rail.

    A warning, not a failure: some literals are the attack itself (``/etc/passwd``, a metadata IP).
    It is shown to the grader, which judges whether the rail generalizes, and recorded so a run can
    show how often rails are fitted to one attack's exact values.
    """
    return sorted(lit for lit in literals if len(lit) >= MIN_LITERAL and lit in instructions)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for item in value.values() for s in _strings(item)]
    if isinstance(value, list | tuple):
        return [s for item in value for s in _strings(item)]
    return []
