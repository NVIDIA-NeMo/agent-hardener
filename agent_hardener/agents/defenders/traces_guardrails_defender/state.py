# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The LangGraph state of one traces guardrails defender run (one attack)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent_hardener.models import DefenderInput, DefenderOutput

from .schemas import BenignToolCall, CandidateCall, Outcome, RailDraft, RailGrade, VulnerabilityAnalysis


class DefenderState(BaseModel):
    """Everything the nodes read and write. Each node returns only the fields it changes."""

    input: DefenderInput
    #: Stable, filesystem-safe name for this attack, used for artifacts and traces.
    attack: str

    # prepare
    rendered_trace: str = ""
    candidates: list[CandidateCall] = Field(default_factory=list)

    # vulnerability_analyzer
    analysis_attempts: int = 0
    #: Why the last analysis could not be grounded in the trace; fed back on the retry.
    grounding_error: str | None = None
    vulnerability_analysis: VulnerabilityAnalysis | None = None

    # benign_args + guardrail_generator
    #: Legitimate calls to the guarded tool (tool route only), shown to the suggestor and the grader.
    benign_calls: list[BenignToolCall] = Field(default_factory=list)
    #: Every suggestor draft and grader verdict, in order; ``rail`` is the best-graded draft.
    drafts: list[RailDraft] = Field(default_factory=list)
    grades: list[RailGrade] = Field(default_factory=list)
    rail: RailDraft | None = None

    # component_writer / skip
    outcome: Outcome | None = None
    output: DefenderOutput | None = None
