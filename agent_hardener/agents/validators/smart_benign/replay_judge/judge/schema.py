# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the refusal judge."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel


class JudgeVerdict(AgentHardenerModel):
    """Typed output the judge must return.

    Schema validation is enforced by ``with_structured_output``; the
    ``reasoning`` floor (≥3 chars) makes "one-letter" reasoning impossible.
    """

    is_refused: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=3)
