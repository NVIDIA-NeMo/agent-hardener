# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the gap_detector's LLM boundary pass."""

from __future__ import annotations

from agent_hardener.models import AgentHardenerModel


class BoundaryVerdict(AgentHardenerModel):
    """One tool's benign/attack-boundary verdict from the LLM.

    Emitted per tool by ``analyze_boundaries``. When ``needs_clarification`` is
    true, ``question`` carries the text to surface in the interview — either an
    abstract boundary question or one or more concrete candidate edge-case
    requests for the user to judge.
    """

    needs_clarification: bool
    question: str = ""
