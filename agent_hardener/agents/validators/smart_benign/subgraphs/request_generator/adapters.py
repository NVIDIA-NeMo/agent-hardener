# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the request_generator agent's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import RequestGeneratorState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def entry_adapter(parent: SynthState) -> RequestGeneratorState:
    """Pull the unified profile out of the parent state.

    Generator knobs (per-tool count, borderline / negative-control counts,
    concurrency) come from the package-level ``defaults.yaml``; per-session
    overrides are out of scope for v1.

    Borderline rows are gated on ``interactive``: they probe a benign/attack
    boundary that is only confirmed via the interview, so non-interactive runs
    (no TTY) generate zero of them rather than replaying an unconfirmed line.
    """
    if parent.profile is None:
        msg = "request_generator entered without a unified profile"
        raise RuntimeError(msg)
    d = defaults()
    return RequestGeneratorState(
        target_name=parent.profile.target_name,
        tools=list(parent.profile.tools),
        personas=list(parent.profile.personas),
        system_role=parent.profile.system_role,
        out_of_scope=list(parent.profile.out_of_scope),
        requests_per_tool=d.generation.requests_per_tool,
        borderline_per_tool=d.generation.borderline_per_tool if parent.interactive else 0,
        negative_control_per_tool=d.generation.negative_control_per_tool,
        concurrency=d.synth_llm.concurrency,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: RequestGeneratorState) -> dict[str, object]:
    """Write the generated requests into ``state.requests`` (single-writer)."""
    delta: dict[str, object] = {"requests": sub.generated}
    if sub.source_note:
        delta["source_notes"] = {"request_generator": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
