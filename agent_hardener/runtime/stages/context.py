# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-process values threaded through one iteration's pipeline stages.

These are execution context, not serialized contracts (unlike the models in
:mod:`agent_hardener.models`), so they are plain frozen dataclasses — same choice as
:class:`~agent_hardener.runtime.run_context.RunContext`. :class:`IterationContext` carries the ambient
per-iteration state every stage needs (recorder, iteration number, the round's attacks, the
iteration/run directories, and the previous iteration's validator feedback); :class:`DefenseResult`
is the defense stage's typed output that the deploy/victim/validation stages consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from agent_hardener.models import AttackRecord, DefenderAnalysis, ValidationFeedback
    from agent_hardener.runtime.round_store import RoundStore
    from agent_hardener.runtime.run_recorder import RunRecorder


@dataclass(frozen=True, slots=True)
class IterationContext:
    """Ambient state for one defend → deploy → victim → validate iteration."""

    recorder: RunRecorder
    store: RoundStore
    iteration: int
    attacks: list[AttackRecord]
    iteration_dir: Path | None
    run_dir: Path | None
    validation_feedback: dict[str, ValidationFeedback] | None = None


@dataclass(frozen=True, slots=True)
class DefenseResult:
    """The defense stage's analyses plus their aggregated policy patches."""

    analyses: list[DefenderAnalysis]
    policy_patches: list[dict[str, Any]]
