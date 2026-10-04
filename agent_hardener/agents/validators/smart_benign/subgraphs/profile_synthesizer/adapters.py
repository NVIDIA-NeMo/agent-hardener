# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the profile_synthesizer agent's private state."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import ProfileSynthesizerState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthInputs, SynthState


def entry_adapter(parent: SynthState) -> ProfileSynthesizerState:
    """Project the parent state into the synthesizer's private inputs."""
    d = defaults()
    return ProfileSynthesizerState(
        target_name=parent.inputs.target_name,
        partials=dict(parent.partials),
        interviewer_answers=list(parent.interviewer_answers),
        source_notes=dict(parent.source_notes),
        input_hash=compute_input_hash(parent.inputs),
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: ProfileSynthesizerState) -> dict[str, object]:
    """Surface the unified profile and any synthesizer notes back to the parent."""
    delta: dict[str, object] = {}
    if sub.profile is not None:
        delta["profile"] = sub.profile
    if sub.source_note:
        delta["source_notes"] = {"synthesizer": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta


def compute_input_hash(inputs: SynthInputs) -> str:
    """SHA-256 (first 16 hex chars) over a canonical JSON dump of the inputs.

    Used as the profile's ``input_hash`` for cache validation. Stable across
    the interview loop because :class:`SynthInputs` does not include the
    runtime-accumulated ``interviewer_answers`` field on ``SynthState``.
    """
    canonical = json.dumps(inputs.model_dump(mode="json"), sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
