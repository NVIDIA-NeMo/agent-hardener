# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What each phase of the traces guardrails defender produces.

``*Response`` models are what an LLM is asked to return. They stay flat and string-typed because
small models fill flat schemas far more reliably than ones with free-form ``dict`` fields. The other
models are what the defender derives from those responses after checking them against the trace.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

#: Where the analysis found the harm: in a tool call (a tool rail is written for it), or only in the user's
#: message (no tool call carried it, so the defender skips the attack).
Route = Literal["tool", "input"]


class CandidateCall(BaseModel):
    """A tool call from the attack trace that could have carried the harm."""

    #: 1-based position in ``AgentTrace.steps``, the same numbering ``render_trace`` prints.
    step: int
    tool_name: str
    args: Any = None
    #: How the call ended in the attack's trace: ``ok`` (ran), ``error`` or ``network_denied``.
    outcome: str = "ok"


class AnalysisResponse(BaseModel):
    """The analyzer LLM's answer."""

    found: bool = Field(description="True if one tool call in the trace executed the harm.")
    step: int | None = Field(default=None, description="The step number of that tool call, as printed in the trace.")
    tool_name: str | None = Field(default=None, description="The name of the tool at that step.")
    malicious_arg_names: list[str] = Field(
        default_factory=list, description="Names of the arguments of that call whose values carry the harm."
    )
    attack_pattern: str = Field(
        default="", description="What makes this attack malicious, generalized beyond the literal values used."
    )
    benign_contrast: str = Field(
        default="", description="What ordinary, legitimate use (of the same tool, or of the agent) looks like."
    )
    evidence: str = Field(default="", description="Why this call, and not another, executed the harm.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    argument_separable: bool = Field(
        default=True,
        description=(
            "True if the call's argument values alone tell this attack from the tool's ordinary use; false if "
            "ordinary requests make the same kind of call and the harm lies in who asked or why."
        ),
    )
    distinguishing_feature: str = Field(
        default="",
        description=(
            "The argument property that marks the attack (for example a path outside the workspace, a shell "
            "metacharacter, an internal host); or, when not separable, what the difference is instead."
        ),
    )


class VulnerabilityAnalysis(BaseModel):
    """Where the attack did its harm, grounded in the trace.

    ``target="tool"``: the call (and the argument values) that executed the harm, values copied from
    the trace rather than from the LLM, so they are exactly what the agent sent.
    ``target="input"``: no tool call carried the harm (``fallback_reason`` says why); the attack is skipped.
    """

    target: Route
    tool_name: str | None = None
    #: The step in ``AgentTrace.steps`` (1-based) of the harmful call.
    call_index: int | None = None
    malicious_args: dict[str, Any] = Field(default_factory=dict)
    call_args: Any = None
    attack_pattern: str = ""
    benign_contrast: str = ""
    evidence: str = ""
    confidence: float = 0.0
    fallback_reason: str | None = None


class RailResponse(BaseModel):
    """The suggestor LLM's answer.

    Named ``attack_description`` rather than ``system_instructions`` (its name in the plugin config):
    asked for "system instructions", small models return their own system prompt.
    """

    attack_description: str = Field(
        description="The attack description the guard model scores against. Only the description itself, "
        "not these instructions."
    )
    rationale: str = Field(default="", description="Why this description blocks the attack but not normal use.")


class GradeResponse(BaseModel):
    """The grader LLM's answer: one 1-5 score per criterion, and what to fix."""

    coverage: int = Field(ge=1, le=5, description="Blocks the analyzed attack.")
    generalization: int = Field(ge=1, le=5, description="Covers the attack class, not only this instance.")
    benign_preservation: int = Field(ge=1, le=5, description="Ordinary use still passes.")
    enforceability: int = Field(ge=1, le=5, description="Decidable from what the guard model sees.")
    specificity: int = Field(ge=1, le=5, description="Concrete, checkable conditions.")
    required_fixes: str = Field(
        default="", description="What the next draft must change, one fix per line. Empty if nothing."
    )


#: The grader's criteria, in the order the rubric lists them.
CRITERIA = ("coverage", "generalization", "benign_preservation", "enforceability", "specificity")


class RailDraft(BaseModel):
    """One rail the suggestor wrote."""

    #: The guarded tool.
    target_tool: str
    system_instructions: str
    rationale: str = ""
    #: Suggestor-grader iteration that produced it (1-based).
    iteration: int
    #: Literal attack values copied into the instructions: a sign the rail is fitted to one attack.
    lint_warnings: list[str] = Field(default_factory=list)
    #: Answers discarded because they repeated the suggestor's own prompt instead of writing a rail.
    echo_retries: int = 0


class RailGrade(BaseModel):
    """The grader's verdict on one draft."""

    iteration: int
    criteria: dict[str, int]
    #: Out of 10: the mean criterion score scaled from 1-5.
    score: float
    required_fixes: list[str] = Field(default_factory=list)


class BenignCallResponse(BaseModel):
    """One benign call the LLM wrote for a benign request."""

    request: str = Field(description="The benign request this call serves, copied exactly.")
    args_json: str = Field(description="The call's arguments as a JSON object, using the tool's argument names.")


class BenignArgsResponse(BaseModel):
    """The benign-args LLM's answer."""

    calls: list[BenignCallResponse]


class BenignToolCall(BaseModel):
    """A legitimate call to the guarded tool: what the rail must keep allowing."""

    tool: str
    args: dict[str, Any]
    request: str


OutcomeStatus = Literal["applied", "skipped", "error"]


class Outcome(BaseModel):
    """What the defender did for one attack: the one file to read when comparing runs."""

    attack: str
    status: OutcomeStatus
    reason: str = ""
    route: Route | None = None
    fallback_reason: str | None = None
    tool_name: str | None = None
    guardrail_name: str | None = None
    #: Score (out of 10) of the applied draft, ``None`` when the grader could not grade it.
    grader_score: float | None = None
    grader_iterations: int = 0
    lint_warnings: list[str] = Field(default_factory=list)
