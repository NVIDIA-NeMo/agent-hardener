# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the relay-instrumentation preflight.

The behaviour under test is a refusal, at two moments. An uninstrumented victim must stop the run
*before* the first attack, because it would otherwise answer every attack and produce a report
claiming a guardrail worked when none was ever installed. A victim whose tool calls bypass Relay must
stop it *after* the first attacks — which is the earliest evidence that exists.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_hardener.errors import VictimNotInstrumentedError, VictimUnavailableError
from agent_hardener.preflight.relay import check_relay_instrumentation, check_tool_path

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "atof" / "concurrent_tool_calls.atof.jsonl"


def _probe_writing(path: Path, source: Path):
    """A victim that emits telemetry when probed."""

    async def probe() -> None:
        path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    return probe


async def _silent_probe() -> None:
    """A victim that answers but emits nothing — the failure this preflight exists to catch."""


def test_instrumented_victim_passes_and_reports_its_tools(tmp_path: Path) -> None:
    atof = tmp_path / "events.atof.jsonl"
    result = asyncio.run(
        check_relay_instrumentation(
            probe=_probe_writing(atof, FIXTURE), atof_path=atof, timeout_seconds=5, poll_interval_seconds=0.01
        )
    )
    assert result.records_observed == 24
    assert result.tools_seen == ("transfer_funds",)
    assert result.quarantined is False


def test_silent_victim_fails_the_run(tmp_path: Path) -> None:
    atof = tmp_path / "events.atof.jsonl"
    with pytest.raises(VictimNotInstrumentedError) as excinfo:
        asyncio.run(
            check_relay_instrumentation(
                probe=_silent_probe, atof_path=atof, timeout_seconds=0.05, poll_interval_seconds=0.01
            )
        )
    assert "never created" in str(excinfo.value)
    assert excinfo.value.category == "victim_not_instrumented"


def test_empty_atof_is_diagnosed_as_a_disabled_sink(tmp_path: Path) -> None:
    """`AtofConfig.enabled` defaults to false, so an empty file is the likeliest misconfiguration."""
    atof = tmp_path / "events.atof.jsonl"
    atof.write_text("", encoding="utf-8")
    with pytest.raises(VictimNotInstrumentedError, match="sink is disabled"):
        asyncio.run(
            check_relay_instrumentation(
                probe=_silent_probe, atof_path=atof, timeout_seconds=0.05, poll_interval_seconds=0.01
            )
        )


def test_pre_existing_records_are_not_counted_as_this_probe(tmp_path: Path) -> None:
    """A mid-run re-check must not pass on an earlier round's telemetry."""
    atof = tmp_path / "events.atof.jsonl"
    atof.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(VictimNotInstrumentedError, match="went untraced"):
        asyncio.run(
            check_relay_instrumentation(
                probe=_silent_probe, atof_path=atof, timeout_seconds=0.05, poll_interval_seconds=0.01
            )
        )


def test_only_new_records_are_reported(tmp_path: Path) -> None:
    """With an earlier round already on disk, the count reflects this probe alone."""
    atof = tmp_path / "events.atof.jsonl"
    existing = json.dumps({"atof_version": "0.1", "kind": "scope", "uuid": "old", "scope_category": "start"})
    atof.write_text(existing + "\n", encoding="utf-8")

    async def probe() -> None:
        with atof.open("a", encoding="utf-8") as handle:
            handle.write(FIXTURE.read_text(encoding="utf-8"))

    result = asyncio.run(
        check_relay_instrumentation(probe=probe, atof_path=atof, timeout_seconds=5, poll_interval_seconds=0.01)
    )
    assert result.records_observed == 24


def test_quarantined_scope_stack_is_surfaced_without_failing(tmp_path: Path) -> None:
    """A dirty scope stack degrades attribution but the victim is still instrumented — report, do not abort."""
    atof = tmp_path / "events.atof.jsonl"

    async def probe() -> None:
        atof.write_text(
            json.dumps(
                {
                    "atof_version": "0.1",
                    "kind": "mark",
                    "metadata": {"note": "an earlier turn left the scope stack dirty"},
                }
            ),
            encoding="utf-8",
        )

    result = asyncio.run(
        check_relay_instrumentation(probe=probe, atof_path=atof, timeout_seconds=5, poll_interval_seconds=0.01)
    )
    assert result.quarantined is True


def test_no_tool_call_across_a_whole_round_is_fatal() -> None:
    """Relay attached, but tool calls bypassing it: unguardable, and the run must not proceed.

    A hand-built LangGraph graph does exactly this — LLM scopes appear, tool calls do not — so the
    startup probe passes and every guardrail the defenders write is inert. Failing loudly beats a
    report that reads as weak defenders while attribution silently falls back to guessing the tool.
    """
    llm_only = [
        {"atof_version": "0.1", "kind": "scope", "scope_category": "start", "category": "llm", "uuid": "u1"},
        {"atof_version": "0.1", "kind": "scope", "scope_category": "end", "category": "llm", "uuid": "u1"},
    ]

    with pytest.raises(VictimNotInstrumentedError, match="do not pass through Relay"):
        check_tool_path(llm_only)


def test_an_observed_tool_call_satisfies_the_tool_path_check() -> None:
    common = {"atof_version": "0.1", "kind": "scope", "category": "tool", "uuid": "u1"}
    records = [
        {**common, "scope_category": "start", "name": "transfer_funds", "data": {}},
        {**common, "scope_category": "end", "data": None},
    ]

    assert check_tool_path(records) == ("transfer_funds",)


def test_a_previous_runs_copy_does_not_seed_the_baseline(tmp_path: Path) -> None:
    """A fetched copy is stale state, not a floor.

    A fresh victim starts a new stream — Fabric writes it to a new runtime-scoped path entirely — so
    a longer file from last run would make a correctly-instrumented victim look silent forever.
    """
    atof = tmp_path / "events.atof.jsonl"
    atof.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")  # last run: many records

    def record(uuid: str) -> str:
        return json.dumps({"atof_version": "0.1", "kind": "scope", "uuid": uuid, "scope_category": "start"})

    pulls = 0

    def sync() -> None:
        # The fresh stream is one record at baseline and two once the probe has been handled — far
        # shorter than last run's file, which is the point.
        nonlocal pulls
        pulls += 1
        lines = [record("new-1")] if pulls == 1 else [record("new-1"), record("new-2")]
        atof.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = asyncio.run(
        check_relay_instrumentation(
            probe=_silent_probe, atof_path=atof, sync=sync, timeout_seconds=0.5, poll_interval_seconds=0.01
        )
    )

    assert result.records_observed == 1  # measured against the new stream, not the old file


def test_a_rejected_probe_blames_the_victim_not_relay(tmp_path: Path) -> None:
    """A dead agent answers 5xx and emits nothing, exactly like an uninstrumented victim.

    Blaming Relay for that silence sends the reader after a bug they do not have — taken from a real
    run where a 503 was reported as missing telemetry.
    """

    class _Rejected:
        status_code = 503
        text = "Fabric runtime startup failed: adapter lifecycle start failed"

    async def probe() -> object:
        return _Rejected()

    with pytest.raises(VictimUnavailableError) as excinfo:
        asyncio.run(
            check_relay_instrumentation(
                probe=probe, atof_path=tmp_path / "events.atof.jsonl", timeout_seconds=0.05, poll_interval_seconds=0.01
            )
        )

    assert "HTTP 503" in str(excinfo.value)
    assert "adapter lifecycle start failed" in str(excinfo.value)
    assert excinfo.value.category == "victim_unavailable"
