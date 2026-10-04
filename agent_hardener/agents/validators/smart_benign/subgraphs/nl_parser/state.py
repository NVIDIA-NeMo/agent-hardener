# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the nl_parser subgraph."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import Persona, ToolSpec
from agent_hardener.models import AgentHardenerModel


class NLParserState(AgentHardenerModel):
    """Private state for the nl_parser subgraph.

    Consumes a free-form ``description`` from the parent state, invokes one
    LangChain ``with_structured_output`` call inside ``nodes/parse.py``, and
    surfaces typed ``ToolSpec`` / ``Persona`` entries tagged ``source="nl"``.
    """

    # --- Inputs (populated by entry_adapter) ---
    description: str | None = None
    target_name: str
    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Outputs (read by exit_adapter) ---
    tools: list[ToolSpec] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    system_role: str | None = None
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
