# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the declarative run-layout catalog + resolver."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from agent_hardener.swarm_tracker import GARAK, OPENSHELL_LOGS, RunLayout, Scope

if TYPE_CHECKING:
    from pathlib import Path


def test_run_scoped_kind_resolves_under_run_dir(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run-1")
    got = layout.dir_for(OPENSHELL_LOGS)
    assert got == tmp_path / "run-logs" / "run-1" / "openshell-logs"
    assert got.is_dir()  # dir_for creates a ready directory


def test_round_scoped_kind_resolves_under_round_dir(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run-1")
    got = layout.dir_for(GARAK, round_n=1)  # attackers run once per round → garak is round-scoped
    assert got == tmp_path / "run-logs" / "run-1" / "round_1" / "garak"
    assert got.is_dir()


def test_base_resolves_each_scope(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run-1")
    assert layout.base(Scope.RUN) == tmp_path / "run-logs" / "run-1"
    assert layout.base(Scope.ROUND, round_n=3) == tmp_path / "run-logs" / "run-1" / "round_3"
    assert layout.base(Scope.ITERATION, round_n=1, iteration_n=2) == (
        tmp_path / "run-logs" / "run-1" / "round_1" / "iteration-2"
    )
    assert layout.base(Scope.TARGET, target="acme") == tmp_path / "benign_profiles" / "acme"
    assert layout.base(Scope.GLOBAL).name == ".agent-hardener"


def test_scoped_bases_require_their_context(tmp_path: Path) -> None:
    layout = RunLayout(tmp_path, "run-1")
    with pytest.raises(ValueError, match="ITERATION"):
        layout.base(Scope.ITERATION)  # missing round_n/iteration_n
    with pytest.raises(ValueError, match="ROUND"):
        layout.base(Scope.ROUND)
    with pytest.raises(ValueError, match="TARGET"):
        layout.base(Scope.TARGET)
