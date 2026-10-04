# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed output schema for the api_prober extraction LLM call."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

# Hard cap on per-tool confidence. ``api_probe`` is the lowest-trust source —
# the merge in ``profile_synthesizer`` weights structural sources higher, and
# this cap reinforces that at the schema level (the extractor LLM can't claim
# confidence > 0.5 even if the responses sound certain).
API_PROBE_MAX_CONFIDENCE = 0.5


class ProberToolHypothesis(AgentHardenerModel):
    """One tool the extractor LLM thinks the target exposes."""

    name: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    example_inputs: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)


class ProberOutput(AgentHardenerModel):
    """Structured extraction of the probe Q&A transcript."""

    tools: list[ProberToolHypothesis] = Field(default_factory=list)
    system_role: str | None = None
    out_of_scope: list[str] = Field(default_factory=list)
