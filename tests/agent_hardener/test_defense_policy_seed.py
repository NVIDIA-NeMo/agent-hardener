# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DefenseStage seeds the policy defender's current_policy from the run's initial policy."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from agent_hardener.runtime.stages.defense import DefenseStage

if TYPE_CHECKING:
    from pathlib import Path


def _stage(policy_path: Path | None) -> DefenseStage:
    # DefenseStage only reads config.storage.victim_policy_path for the seed, so a light stand-in suffices.
    config = SimpleNamespace(storage=SimpleNamespace(victim_policy_path=policy_path))
    return DefenseStage(config, invoker=None)  # type: ignore[arg-type]


def test_policy_seed_prefers_run_active_state(tmp_path: Path) -> None:
    initial = tmp_path / "policy.yaml"
    initial.write_text("template: true\n", encoding="utf-8")
    run_dir = tmp_path / "run-logs" / "r1"
    active = run_dir / "victim-active-state" / "policy.yaml"
    active.parent.mkdir(parents=True)
    active.write_text("inferred: true\n", encoding="utf-8")

    seed = _stage(initial)._policy_seed(run_dir)

    assert seed == active  # the per-run active-state copy wins over the configured seed


def test_policy_seed_falls_back_to_configured_policy(tmp_path: Path) -> None:
    initial = tmp_path / "policy.yaml"
    initial.write_text("inferred: true\n", encoding="utf-8")

    # No run_dir (standalone) → the configured (inferred) policy is used.
    assert _stage(initial)._policy_seed(None) == initial
    # run_dir present but no active-state copy yet → still falls back to the configured policy.
    assert _stage(initial)._policy_seed(tmp_path / "run-logs" / "r1") == initial


def test_policy_seed_is_none_without_a_policy_file(tmp_path: Path) -> None:
    assert _stage(None)._policy_seed(None) is None
    assert _stage(tmp_path / "missing.yaml")._policy_seed(None) is None
