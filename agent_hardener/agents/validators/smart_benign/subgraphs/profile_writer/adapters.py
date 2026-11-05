# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapters bridging SynthState and the profile_writer agent's private state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .state import ProfileWriterState

if TYPE_CHECKING:
    from agent_hardener.agents.validators.smart_benign.state import SynthState


def target_dir(parent: SynthState) -> Path:
    """Where the synth artifacts go.

    ``parent.artifact_dir`` if the top-level validator set it
    (``storage.root_dir/benign_profiles/<target_name>/``); otherwise
    ``cwd/benign_profiles/<target_name>/`` for standalone CLI use.
    """
    return parent.artifact_dir or (Path.cwd() / "benign_profiles" / parent.inputs.target_name)


def entry_adapter(parent: SynthState) -> ProfileWriterState:
    """Pull the profile, requests, notes, and interview record; derive the target directory."""
    return ProfileWriterState(
        target_dir=target_dir(parent),
        profile=parent.profile,
        requests=list(parent.requests),
        source_notes=dict(parent.source_notes),
        source_details=dict(parent.source_details),
        interview_log=list(parent.interview_log),
        pre_interview_profile=parent.pre_interview_profile,
    )


def exit_adapter(sub: ProfileWriterState) -> dict[str, object]:
    """Surface only the writer's own note + errors back to the parent."""
    delta: dict[str, object] = {}
    if sub.source_note:
        delta["source_notes"] = {"profile_writer": sub.source_note}
    if sub.errors:
        delta["errors"] = sub.errors
    return delta
