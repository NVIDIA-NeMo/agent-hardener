# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import runpy
from typing import TYPE_CHECKING

import pytest

from agent_hardener.final_log import cli

if TYPE_CHECKING:
    from pathlib import Path

_NO_REPORTS = "No report.json artifacts found for this run."


def _write_report(root: Path, mission: str, *, mtime: float | None = None) -> Path:
    path = root / mission / "report.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"round_id": f"round-{mission}", "mission_id": mission, "success": True}))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_find_report_paths_returns_sorted_reports_under_the_root(tmp_path: Path) -> None:
    b, a = _write_report(tmp_path, "b"), _write_report(tmp_path, "a")
    (tmp_path / "a" / "other.json").write_text("{}")

    assert cli.find_report_paths(root=tmp_path) == [a, b]


def test_find_report_paths_is_empty_when_the_root_is_missing(tmp_path: Path) -> None:
    assert cli.find_report_paths(root=tmp_path / "absent") == []


def test_find_report_paths_filters_by_mission_id_substring(tmp_path: Path) -> None:
    _write_report(tmp_path, "mission-1")
    wanted = _write_report(tmp_path, "mission-2")

    assert cli.find_report_paths(root=tmp_path, mission_ids=["mission-2"]) == [wanted]
    assert len(cli.find_report_paths(root=tmp_path, mission_ids=[])) == 2


def test_find_report_paths_drops_reports_older_than_the_marker(tmp_path: Path) -> None:
    _write_report(tmp_path, "old", mtime=1_000)
    fresh = _write_report(tmp_path, "fresh", mtime=3_000)
    marker = tmp_path / "marker"
    marker.touch()
    os.utime(marker, (2_000, 2_000))

    assert cli.find_report_paths(root=tmp_path, marker_path=marker) == [fresh]


def test_find_report_paths_ignores_a_marker_that_does_not_exist(tmp_path: Path) -> None:
    only = _write_report(tmp_path, "m", mtime=1_000)

    assert cli.find_report_paths(root=tmp_path, marker_path=tmp_path / "missing") == [only]


def test_render_from_paths_tolerates_an_unreadable_report(tmp_path: Path) -> None:
    good = _write_report(tmp_path, "good")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")

    text = cli.render_final_run_log_from_paths([good, broken])

    assert text.startswith("Agent Hardener final log\n")
    assert "round-good: success=yes" in text
    assert "(1 succeeded, 1 failed)" in text


def test_main_reports_nothing_found_and_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--root", str(tmp_path / "empty")]) == 0

    out = capsys.readouterr().out
    assert out.splitlines()[0] == "Agent Hardener final log"
    assert _NO_REPORTS in out


def test_main_renders_discovered_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _write_report(tmp_path, "m1")

    assert cli.main(["--root", str(tmp_path)]) == 0

    out = capsys.readouterr().out
    assert "Agent Hardener final log" in out
    assert _NO_REPORTS not in out


def test_main_explicit_report_flags_bypass_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    explicit = _write_report(tmp_path / "elsewhere", "chosen")
    monkeypatch.setattr(cli, "find_report_paths", lambda **_: pytest.fail("discovery must not run"))

    assert cli.main(["--report", str(explicit), "--root", str(tmp_path / "ignored")]) == 0

    assert _NO_REPORTS not in capsys.readouterr().out


def test_main_renders_only_the_requested_missions(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    for mission in ("m1", "m2", "m3"):
        _write_report(tmp_path, mission)

    assert cli.main(["--root", str(tmp_path), "--mission-id", "m1", "--mission-id", "m3"]) == 0

    out = capsys.readouterr().out
    assert "round-m1" in out
    assert "round-m3" in out
    assert "round-m2" not in out
    assert _NO_REPORTS not in out


def test_module_entry_point_exits_with_the_cli_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["final_log", "--root", str(tmp_path / "empty")])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("agent_hardener.final_log", run_name="__main__")

    assert exit_info.value.code == 0
    assert _NO_REPORTS in capsys.readouterr().out
