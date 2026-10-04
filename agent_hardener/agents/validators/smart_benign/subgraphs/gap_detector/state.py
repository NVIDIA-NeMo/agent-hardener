# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the gap_detector subgraph."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import ToolSpec, VictimCapabilityProfile
from agent_hardener.models import AgentHardenerModel


class GapDetectorState(AgentHardenerModel):
    """Private state for the gap_detector subgraph.

    Reads the merged ``profile`` and per-source ``partials`` from the parent
    ``SynthState`` and emits a list of human-readable gap descriptors. The
    deterministic ``detect_gaps`` rules run first; while the interview is open
    (``interview_open``), an LLM ``analyze_boundaries`` pass appends
    ``boundary[<tool>]:`` gaps about each tool's benign/attack line.
    """

    profile: VictimCapabilityProfile | None = None
    partials: dict[str, list[ToolSpec]] = Field(default_factory=dict)
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    gaps: list[str] = Field(default_factory=list)

    # --- Boundary pass (LLM, open-interview-only) ---
    # False on the closing pass (round/question budget spent, or non-interactive)
    # so the per-tool boundary fan-out is skipped when its questions can no
    # longer be asked.
    interview_open: bool = False
    interviewer_answers: list[tuple[str, str, str]] = Field(default_factory=list)
    concurrency: int = Field(default=2, ge=1)
    synth_model: str = ""
    synth_base_url: str = ""
    synth_api_key: str = ""
