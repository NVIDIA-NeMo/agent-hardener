# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent LangGraph state for the smart benign validator's synth DAG.

The pipeline operates on a single :class:`SynthState` instance that flows through
the parent graph. Ingestion sub-graphs run in parallel and contribute to
``partials`` / ``source_notes`` / ``errors`` via standard LangGraph reducers;
the synthesizer, gap_detector, interviewer, request_generator, and critic
mutate the single-writer fields. Subgraph-internal state is private — see
each subgraph's own ``state.py``.
"""

from __future__ import annotations

from operator import add, or_
from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from agent_hardener.models import AgentHardenerModel

from .models import GeneratedRequest, ToolSpec, VictimCapabilityProfile


class SynthInputs(AgentHardenerModel):
    """User-supplied inputs that drive the synth DAG.

    Any combination of fields may be provided; the DAG branches on what is
    present (e.g., ``github_analyzer`` only runs if ``github_url`` is set).
    """

    target_name: str = Field(min_length=1)
    description: str | None = None
    github_url: str | None = None
    api_endpoint: str | None = None
    # Content hash of the victim's workflow config (when known). Folded into the
    # cache key so a workflow change re-synthesizes even when the endpoint is
    # unchanged — api-probe-only sourcing otherwise can't tell the agent changed.
    victim_workflow_hash: str | None = None
    interview_answers: list[tuple[str, str]] = Field(default_factory=list)
    skip_nl_parser: bool = False
    skip_github_analysis: bool = False
    skip_api_probe: bool = False


class SynthState(AgentHardenerModel):
    """Parent LangGraph state passed between subgraph nodes.

    Fields fall into three buckets:

    1. **Inputs / config** (``inputs``, ``interactive``) — set once at entry.
    2. **Multi-writer accumulators** (``partials``, ``source_notes``,
       ``errors``, ``interviewer_answers``) — typed with prebuilt LangGraph
       reducers (``operator.add`` for lists, ``operator.or_`` for dict merges)
       so parallel subgraphs can contribute without stomping each other.
    3. **Single-writer fields** (``profile``, ``gaps``, ``requests``,
       ``interview_iteration``) — one subgraph owns each.

    The parent graph never reads or writes subgraph-internal state; every
    cross-subgraph value lives here.
    """

    # --- Inputs / config (stable across the run) ---
    inputs: SynthInputs
    interactive: bool = False
    artifact_dir: Path | None = None

    # --- Multi-writer accumulators (prebuilt reducers) ---
    partials: Annotated[dict[str, list[ToolSpec]], or_] = Field(default_factory=dict)
    source_notes: Annotated[dict[str, str], or_] = Field(default_factory=dict)
    errors: Annotated[list[str], add] = Field(default_factory=list)
    interviewer_answers: Annotated[list[tuple[str, str, str]], add] = Field(default_factory=list)

    # --- Inputs / config (set once at entry) ---
    max_interview_questions: int = Field(default=10, ge=1)
    max_interview_rounds: int = Field(default=2, ge=1)

    # --- Single-writer fields ---
    profile: VictimCapabilityProfile | None = None
    gaps: list[str] = Field(default_factory=list)
    requests: list[GeneratedRequest] = Field(default_factory=list)
    interview_questions_asked: int = Field(default=0, ge=0)
    interview_rounds: int = Field(default=0, ge=0)
    # Serialized questions the compose node stages for the ask node's interrupt; plain dicts so they're JSON-ready.
    composed_questions: list[dict[str, Any]] = Field(default_factory=list)

    @classmethod
    def from_inputs(
        cls, inputs: SynthInputs, *, interactive: bool, artifact_dir: Path | None, **overrides: Any
    ) -> SynthState:
        """Build the initial synth state, seeding ``interviewer_answers`` from pre-supplied answers.

        The single construction site for the parent state: both the orchestrator pre-flight
        (``validator._run_synth_dag``) and the ``synth-benign`` wizard go through here. ``overrides`` (e.g.
        ``max_interview_questions``) fall through to the model, so unset knobs keep their field defaults.
        """
        return cls(
            inputs=inputs,
            interactive=interactive,
            artifact_dir=artifact_dir,
            interviewer_answers=[("", question, answer) for question, answer in inputs.interview_answers],
            **overrides,
        )
