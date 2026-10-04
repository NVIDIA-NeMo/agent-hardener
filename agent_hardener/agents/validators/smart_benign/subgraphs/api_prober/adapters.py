# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the api_prober agent's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import ApiProberState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState

PROBE_MODEL_NAME = "smart_benign:api_prober"


def entry_adapter(parent: SynthState) -> ApiProberState:
    """Pull the API endpoint to probe + synth credentials for the extractor."""
    d = defaults()
    return ApiProberState(
        target_url=parent.inputs.api_endpoint or "",
        target_name=parent.inputs.target_name,
        probe_model=PROBE_MODEL_NAME,
        max_probes=0 if parent.inputs.skip_api_probe else d.pipeline.api_probe_budget,
        timeout_seconds=d.replay.timeout_seconds,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: ApiProberState) -> dict[str, object]:
    """Write extracted tools into ``partials["api_probe"]`` on the parent state."""
    delta: dict[str, object] = {}
    if sub.tools:
        delta["partials"] = {"api_probe": sub.tools}
    if sub.source_note:
        delta["source_notes"] = {"api_probe": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
