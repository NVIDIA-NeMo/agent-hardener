# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private state for the github_analyzer agent."""

from __future__ import annotations

from pydantic import Field

from agent_hardener.agents.validators.smart_benign.models import Persona, ToolSpec
from agent_hardener.models import AgentHardenerModel


class GitHubAnalyzerState(AgentHardenerModel):
    """Private state for the github_analyzer agent.

    Shallow-clones the repo into a tmp dir, scans NAT-style workflow YAMLs
    for tool declarations (deterministic, high-confidence), then runs one
    LLM call over the README to extract ``system_role`` / ``personas`` /
    ``out_of_scope``. The tmp dir is created and torn down inside the
    ``clone_and_scan`` node so failures never leak directories.
    """

    # --- Inputs (populated by entry_adapter) ---
    repo_url: str
    target_name: str
    gh_token: str | None = None
    clone_timeout_seconds: float = Field(default=120.0, gt=0)
    synth_model: str
    synth_base_url: str
    synth_api_key: str

    # --- Working state (set by clone_and_scan) ---
    head_sha: str | None = None
    readme_text: str | None = None
    declared_tools: list[ToolSpec] = Field(default_factory=list)
    yaml_files_scanned: int = 0

    # --- Output (read by exit_adapter) ---
    tools: list[ToolSpec] = Field(default_factory=list)
    personas: list[Persona] = Field(default_factory=list)
    system_role: str | None = None
    out_of_scope: list[str] = Field(default_factory=list)
    source_note: str = ""
    errors: list[str] = Field(default_factory=list)
