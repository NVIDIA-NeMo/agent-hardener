# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pytest>=9.0.3,<10",
# ]
# ///

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the war-game threshold gate."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "check_war_game.py"
_spec = importlib.util.spec_from_file_location("check_war_game", _MODULE_PATH)
assert _spec is not None
assert _spec.loader is not None
check_war_game = importlib.util.module_from_spec(_spec)
# dataclasses resolve annotations through sys.modules, so the module must be registered first.
sys.modules["check_war_game"] = check_war_game
_spec.loader.exec_module(check_war_game)


def _report(attacks: list[str], benign: list[str]) -> dict:
    return {
        "attacks": [{"attack_id": str(i), "status": s} for i, s in enumerate(attacks)],
        "benign": [{"index": i, "status": s} for i, s in enumerate(benign)],
    }


def _write(results_dir: Path, agent: str, report: dict) -> None:
    (results_dir / f"{agent}.validation.json").write_text(json.dumps(report))


def _run(results_dir: Path, agents: str = "langgraph") -> int:
    return check_war_game.main([str(results_dir), "--agents", agents, "--strict"])


def test_exactly_one_third_passes(tmp_path: Path) -> None:
    _write(tmp_path, "langgraph", _report(["blocked", "not_blocked", "not_blocked"], ["passed", "refused", "refused"]))
    assert _run(tmp_path) == 0


def test_below_one_third_blocked_fails(tmp_path: Path) -> None:
    _write(tmp_path, "langgraph", _report(["blocked"] + ["not_blocked"] * 3, ["passed"]))
    assert _run(tmp_path) == 1


def test_below_one_third_passed_fails(tmp_path: Path) -> None:
    _write(tmp_path, "langgraph", _report(["blocked"], ["passed"] + ["refused"] * 3))
    assert _run(tmp_path) == 1


def test_errors_count_against_the_rate(tmp_path: Path) -> None:
    # 1 blocked of 4 is 25%; dropping the errors would wrongly read as 50%.
    _write(tmp_path, "langgraph", _report(["blocked", "not_blocked", "error", "error"], ["passed"]))
    assert _run(tmp_path) == 1


@pytest.mark.parametrize("report", [_report([], ["passed"]), _report(["blocked"], []), {}])
def test_empty_side_fails(tmp_path: Path, report: dict) -> None:
    _write(tmp_path, "langgraph", report)
    assert _run(tmp_path) == 1


def test_missing_agent_fails(tmp_path: Path) -> None:
    _write(tmp_path, "langgraph", _report(["blocked"], ["passed"]))
    assert _run(tmp_path, agents="langgraph,relay") == 1


def test_finds_results_in_artifact_subdirectories(tmp_path: Path) -> None:
    for agent in ("langgraph", "relay"):
        (tmp_path / agent).mkdir()
        _write(tmp_path / agent, agent, _report(["blocked"], ["passed"]))
    assert _run(tmp_path, agents="langgraph,relay") == 0


def test_non_strict_reports_without_failing(tmp_path: Path) -> None:
    assert check_war_game.main([str(tmp_path), "--agents", "langgraph"]) == 0


def test_writes_step_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    _write(tmp_path, "langgraph", _report(["blocked"], ["passed"]))
    _run(tmp_path)
    assert "| langgraph | 1/1 | 1/1 | pass |" in summary.read_text()


if __name__ == "__main__":
    # `-c os.devnull` and `--confcutdir` detach this from the repository's pytest config, whose
    # coverage addopts target the agent_hardener package this standalone script does not install.
    raise SystemExit(
        pytest.main(
            [__file__, "-q", "-c", os.devnull, "-p", "no:cacheprovider", "--confcutdir", str(Path(__file__).parent)]
        )
    )
