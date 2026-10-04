# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-round disk facade: the round's directory layout and artifact writes.

:class:`RoundStore` holds ``round_dir`` and wraps the Swarm Tracker disk primitives so round control
flow and the pipeline stages write artifacts without importing ``swarm_tracker`` or threading
``round_dir``. Its sibling :class:`~agent_hardener.runtime.run_recorder.RunRecorder` owns the event stream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_hardener.storage import write_json
from agent_hardener.swarm_tracker import (
    create_iteration_dir,
    snapshot_victim_state,
    write_component_output,
)

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.swarm_tracker import ArtifactKind


class RoundStore:
    """Own one round's directory layout and artifact writes."""

    def __init__(self, round_dir: Path) -> None:
        self.round_dir = round_dir

    def iteration_dir(self, iteration: int) -> Path:
        """Create and return this round's ``iteration-<N>/`` directory."""
        return create_iteration_dir(self.round_dir, iteration)

    def round_artifact_dir(self, kind: ArtifactKind) -> Path:
        """Resolve/create the ready output dir for a ROUND-scoped artifact kind (e.g. GARAK)."""
        artifact_dir = self.round_dir / kind.name
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def write_round_json(self, name: str, obj: Any) -> Path:
        """Write ``obj`` as JSON directly under the round directory and return the path."""
        path = self.round_dir / name
        write_json(path, obj)
        return path

    def write_component(self, target_dir: Path, role: str, name: str, output: Any) -> None:
        """Persist one component's output under ``target_dir``."""
        write_component_output(target_dir, role, name, output)

    def snapshot_victim(self, iteration_dir: Path, policy_src: Path | None, workflow_src: Path | None) -> None:
        """Snapshot the deployed victim agent files into the iteration directory."""
        snapshot_victim_state(iteration_dir, policy_src, workflow_src)
