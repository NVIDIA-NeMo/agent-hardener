# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the github_analyzer agent's private state."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent_hardener.agents.validators.smart_benign.config import defaults
from agent_hardener.env import github_token, inference_api_key

from .state import GitHubAnalyzerState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def entry_adapter(parent: SynthState) -> GitHubAnalyzerState:
    """Pull the repo URL + optional GH_TOKEN + synth credentials."""
    d = defaults()
    return GitHubAnalyzerState(
        repo_url="" if parent.inputs.skip_github_analysis else (parent.inputs.github_url or ""),
        target_name=parent.inputs.target_name,
        gh_token=github_token(),
        synth_model=d.synth_llm.model,
        synth_base_url=d.synth_llm.base_url,
        synth_api_key=inference_api_key() or "",
    )


def exit_adapter(sub: GitHubAnalyzerState) -> dict[str, object]:
    """Write extracted tools into ``partials["github"]`` on the parent state.

    The HEAD SHA is embedded in ``source_notes["github"]`` so the parent's
    cache layer can hash it and invalidate the cached profile when the repo
    changes.
    """
    delta: dict[str, object] = {}
    if sub.tools:
        delta["partials"] = {"github": sub.tools}
    if sub.source_note:
        delta["source_notes"] = {"github": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
