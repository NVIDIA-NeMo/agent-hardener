# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the interviewer LLM call."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel


class InterviewOption(AgentHardenerModel):
    """One selectable answer the LLM proposes for an interview question."""

    label: str
    description: str
    recommended: bool = False


class InterviewerQuestion(AgentHardenerModel):
    """Single question with selectable options."""

    question: str | None = None
    gap: str = ""
    options: list[InterviewOption] = Field(default_factory=list)


class InterviewBatch(AgentHardenerModel):
    """A batch of questions generated in one LLM call."""

    questions: list[InterviewerQuestion] = Field(default_factory=list)
