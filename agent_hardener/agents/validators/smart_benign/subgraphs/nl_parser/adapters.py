# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the nl_parser subgraph's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import inference_api_key

from .state import NLParserState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def entry_adapter(parent: SynthState) -> NLParserState:
    """Pull the description and synth-LLM credentials out of the parent state."""
    d = defaults()
    return NLParserState(
        description=None if parent.inputs.skip_nl_parser else parent.inputs.description,
        target_name=parent.inputs.target_name,
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: NLParserState) -> dict[str, object]:
    """Project nl_parser results into a SynthState delta.

    Only fields with content are included so empty subgraph runs don't write
    empty entries into ``partials`` / ``source_notes``.
    """
    delta: dict[str, object] = {}
    if sub.tools:
        delta["partials"] = {"nl": sub.tools}
    if sub.source_note:
        delta["source_notes"] = {"nl": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
