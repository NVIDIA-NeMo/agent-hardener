# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed loader for the smart benign validator defaults.

The defaults live in ``defaults.yaml`` next to this module so they can be edited
without touching code. This module parses + validates the YAML into a strict
Pydantic model and exposes a cached :func:`defaults` accessor.

Per-session overrides flow through ``AgentConfig.config`` in the session YAML;
nothing here is meant to be mutated at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field

from agent_hardener import llm
from agent_hardener.models import AgentHardenerModel

DEFAULTS_FILE = Path(__file__).parent / "defaults.yaml"


class SynthLLMDefaults(AgentHardenerModel):
    """Defaults for the LLM used by the synth sub-agents.

    ``model``/``base_url`` are optional: when unset in ``defaults.yaml`` they fall back to the shared
    :mod:`agent_hardener.llm` factory default (itself ``AGENT_HARDENER_MODEL``/``AGENT_HARDENER_BASE_URL`` aware),
    so the single "analysis" model lever governs the benign validator alongside the defenders.
    """

    model: str = Field(default_factory=lambda: llm.DEFAULT_MODEL)
    base_url: str = Field(default_factory=lambda: llm.DEFAULT_BASE_URL)
    retries: int = Field(ge=0)
    concurrency: int = Field(ge=1)


class JudgeLLMDefaults(AgentHardenerModel):
    """Defaults for the LLM judge used by replay.

    ``model``/``base_url`` fall back to the shared :mod:`agent_hardener.llm` factory default when unset
    (see :class:`SynthLLMDefaults`).
    """

    model: str = Field(default_factory=lambda: llm.DEFAULT_MODEL)
    base_url: str = Field(default_factory=lambda: llm.DEFAULT_BASE_URL)
    max_tokens: int = Field(gt=0)
    timeout_seconds: float = Field(gt=0)
    temperature: float | None = None
    # Cap on the victim response text fed to the judge, so a verbose answer doesn't inflate judge latency.
    input_chars: int = Field(gt=0)


class GenerationDefaults(AgentHardenerModel):
    """Defaults for the request generator output."""

    requests_per_tool: int = Field(ge=1)
    borderline_per_tool: int = Field(ge=0)
    negative_control_per_tool: int = Field(ge=0)


class ReplayDefaults(AgentHardenerModel):
    """Defaults for the replay + judge stage."""

    timeout_seconds: float = Field(gt=0)
    excerpt_chars: int = Field(ge=0)
    confidence_cutoff: float = Field(ge=0, le=1)
    concurrency: int = Field(ge=1)


class PipelineDefaults(AgentHardenerModel):
    """Defaults that bound pipeline behavior (loop caps, budgets)."""

    max_interview_questions: int = Field(ge=1)
    max_interview_rounds: int = Field(ge=1)
    api_probe_budget: int = Field(ge=0)
    github_llm_call_budget: int = Field(ge=0)


class Defaults(AgentHardenerModel):
    """All smart benign validator defaults, loaded from defaults.yaml."""

    synth_llm: SynthLLMDefaults
    judge_llm: JudgeLLMDefaults
    generation: GenerationDefaults
    replay: ReplayDefaults
    pipeline: PipelineDefaults


@lru_cache(maxsize=1)
def defaults() -> Defaults:
    """Return the parsed defaults, validated against the typed schema."""
    raw = yaml.safe_load(DEFAULTS_FILE.read_text(encoding="utf-8"))
    return Defaults.model_validate(raw)
