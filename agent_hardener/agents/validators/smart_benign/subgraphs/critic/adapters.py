# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the critic agent's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import CriticState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def entry_adapter(parent: SynthState) -> CriticState:
    """Pull the generated requests and the profile's out_of_scope / tool list."""
    d = defaults()
    out_of_scope = list(parent.profile.out_of_scope) if parent.profile else []
    tool_names = [t.name for t in parent.profile.tools] if parent.profile else []
    return CriticState(
        target_name=parent.inputs.target_name,
        requests=list(parent.requests),
        out_of_scope=out_of_scope,
        tool_names=tool_names,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: CriticState) -> dict[str, object]:
    """Replace ``state.requests`` with the filtered list (single-writer field)."""
    delta: dict[str, object] = {"requests": sub.filtered_requests}
    if sub.source_note:
        delta["source_notes"] = {"critic": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
