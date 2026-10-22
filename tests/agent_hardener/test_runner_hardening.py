# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the mission hands the orchestrator: the event bus, ATOF sync and the tool-path check."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from agent_hardener.loggers import emit_event
from agent_hardener.runtime import runner

if TYPE_CHECKING:
    from pathlib import Path

    from conftest import BuildMission

pytestmark = pytest.mark.unit


def test_events_reach_the_http_sink_when_one_is_configured(
    build_mission: BuildMission, monkeypatch: pytest.MonkeyPatch
) -> None:
    received: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        runner, "make_http_event_sink", lambda: lambda event, payload: received.append((event, payload))
    )
    harness = build_mission(no_cleanup=True)
    harness.orchestrator_hook = lambda _kwargs: emit_event("phase_started", {"phase": "attack"})

    harness.run()

    assert ("phase_started", {"phase": "attack"}) in received


# --- tool path -------------------------------------------------------------------------------------------


def test_tool_path_verification_is_skipped_for_a_plain_http_target(build_mission: BuildMission) -> None:
    harness = build_mission(no_cleanup=True)
    harness.orchestrator_hook = lambda kwargs: kwargs["verify_tool_path"]()

    harness.run()

    assert not any("tool calls reach Relay" in line for line in harness.lines)
    assert harness.lifecycle.fetched == []


@pytest.mark.usefixtures("relay_probe")
@pytest.mark.parametrize("atof_exists", [True, False])
def test_tool_path_verification_reads_the_synced_log_and_names_the_tools(
    build_mission: BuildMission, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atof_exists: bool
) -> None:
    atof = tmp_path / "events.atof.jsonl"
    if atof_exists:
        atof.write_text("{}\n")
    monkeypatch.setattr(runner, "load_records", lambda path: [f"record-from-{path.name}"])
    checked: list[list[str]] = []

    def fake_check(records: list[str]) -> list[str]:
        checked.append(records)
        return ["bash", "read_file"]

    monkeypatch.setattr(runner, "check_tool_path", fake_check)
    harness = build_mission(atof_path=atof, no_cleanup=True)
    harness.orchestrator_hook = lambda kwargs: kwargs["verify_tool_path"]()

    harness.run()

    assert checked == [["record-from-events.atof.jsonl"] if atof_exists else []]
    assert "tool calls reach Relay (bash, read_file)" in harness.lines
    assert harness.lifecycle.fetched[-1] == (f"{runner.SANDBOX_ARTIFACTS_DIR}/events.atof.jsonl", atof)


# --- ATOF sync -------------------------------------------------------------------------------------------


def test_atof_sync_is_a_noop_for_a_plain_http_target(build_mission: BuildMission) -> None:
    harness = build_mission(no_cleanup=True)
    harness.orchestrator_hook = lambda kwargs: kwargs["sync_atof"]()

    harness.run()

    assert harness.lifecycle.fetched == []


@pytest.mark.usefixtures("relay_probe")
def test_atof_sync_is_best_effort(build_mission: BuildMission, tmp_path: Path) -> None:
    harness = build_mission(atof_path=tmp_path / "events.atof.jsonl", no_cleanup=True)
    harness.lifecycle.fetch_error = RuntimeError("sandbox gone")
    harness.orchestrator_hook = lambda kwargs: kwargs["sync_atof"]()

    result = harness.run()

    assert result.success
