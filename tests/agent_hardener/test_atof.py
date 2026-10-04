# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for recovering a victim's tool calls from its Relay ATOF log.

The fixture is *real* ATOF, captured from nemo-relay 0.7.3 running six concurrent tool calls whose
completion order (``[5, 1, 3, 4, 2, 0]``) deliberately differs from their start order — so a reader
that pairs a start with an end by position rather than by uuid passes on a serial log and fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_hardener.atof import has_quarantined_scopes, is_atof_record, iter_records, load_records, tool_calls

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "atof" / "concurrent_tool_calls.atof.jsonl"


@pytest.fixture
def records() -> list[dict]:
    return load_records(FIXTURE)


def test_every_concurrent_call_is_recovered_with_its_own_arguments(records: list[dict]) -> None:
    """Six interleaved calls, each start paired with its own end rather than the next one in file."""
    calls = tool_calls(records)
    assert len(calls) == 6
    assert {call.args["n"] for call in calls} == set(range(6))
    assert all(call.args["n"] == call.result["n"] for call in calls)


def test_completion_order_differs_from_start_order(records: list[dict]) -> None:
    """Guards the fixture itself: if it stopped interleaving, the test above would prove nothing."""
    ends = [r for r in records if r.get("scope_category") == "end" and r.get("category") == "tool"]
    completion = [r["data"]["n"] for r in sorted(ends, key=lambda r: r["timestamp"])]
    assert completion != sorted(completion)


def test_tool_calls_carry_args_and_result(records: list[dict]) -> None:
    call = next(c for c in tool_calls(records) if c.args["n"] == 3)
    assert call.name == "transfer_funds"
    assert call.result == {"n": 3}
    assert call.ok is True


def _write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "events.atof.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_torn_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """The victim may still be writing; a half-flushed record must not fail the run."""
    good = json.dumps({"atof_version": "0.1", "kind": "scope", "uuid": "a", "scope_category": "start"})
    path = _write(tmp_path, good, '{"atof_version":"0.1","kind":"sco')
    assert len(load_records(path)) == 1


def test_non_atof_json_is_ignored(tmp_path: Path) -> None:
    """A user may point us at the wrong JSONL; claiming it would invent tool calls."""
    path = _write(tmp_path, json.dumps({"level": "info", "msg": "unrelated log line"}))
    assert load_records(path) == []


def test_end_without_start_is_skipped(tmp_path: Path) -> None:
    """Reading ahead of the writer truncates a pair; an unpaired end has no args to report."""
    end = json.dumps(
        {"atof_version": "0.1", "kind": "scope", "scope_category": "end", "category": "tool", "uuid": "orphan"}
    )
    assert tool_calls(load_records(_write(tmp_path, end))) == []


def test_in_flight_call_is_not_reported(tmp_path: Path) -> None:
    """A tool that has not returned has no outcome, so it must not be attributed yet."""
    start = json.dumps(
        {
            "atof_version": "0.1",
            "kind": "scope",
            "scope_category": "start",
            "category": "tool",
            "uuid": "u1",
            "name": "transfer_funds",
        }
    )
    assert tool_calls(load_records(_write(tmp_path, start))) == []


def test_a_parent_cycle_does_not_hang() -> None:
    """A malformed log must not spin any parent walk forever."""
    scope = {"atof_version": "0.1", "kind": "scope", "scope_category": "start", "category": "tool"}
    records = [
        {**scope, "uuid": "a", "parent_uuid": "b", "name": "t"},
        {**scope, "uuid": "b", "parent_uuid": "a", "name": "t"},
        {**scope, "uuid": "a", "scope_category": "end", "data": None},
    ]
    assert len(tool_calls(records)) == 1


def test_error_status_marks_the_call_failed() -> None:
    """A guardrail rejection surfaces as an error status on the scope end event."""
    common = {"atof_version": "0.1", "kind": "scope", "category": "tool", "uuid": "u1"}
    records = [
        {**common, "scope_category": "start", "name": "transfer_funds", "data": {"amount": 9999}},
        {**common, "scope_category": "end", "metadata": {"status_code": "ERROR"}},
    ]
    (call,) = tool_calls(records)
    assert call.ok is False


def test_quarantined_scope_stack_is_surfaced() -> None:
    """A dirty scope stack degrades the stream silently, so the run must not read as clean."""
    assert has_quarantined_scopes([{"kind": "mark", "metadata": {"note": "… left the scope stack dirty …"}}])
    assert not has_quarantined_scopes([{"kind": "mark", "metadata": {"note": "fine"}}])


def test_is_atof_record_accepts_envelope_without_version() -> None:
    assert is_atof_record({"kind": "scope"})
    assert not is_atof_record({"kind": "not-a-scope"})


def test_iter_records_streams_without_loading_everything() -> None:
    assert next(iter_records(FIXTURE))["atof_version"] == "0.1"


# --- against a real victim's stream --------------------------------------------------------------

REAL_STREAM = Path(__file__).resolve().parents[1] / "fixtures" / "atof" / "langchain_concurrent.atof.jsonl"


def test_a_real_concurrent_langchain_stream_yields_its_tool_calls() -> None:
    """Recorded from the example victim under six interleaved invocations.

    Synthetic fixtures encode what we believe Relay emits; this one is what it actually emitted. It
    is what the relay preflight reads to decide the victim is instrumented, so an empty result here
    is the same signal as a victim with no Relay at all.
    """
    records = load_records(REAL_STREAM)

    assert not has_quarantined_scopes(records)
    assert {call.name for call in tool_calls(records)} == {"transfer_funds", "read_customer_record", "send_email"}
