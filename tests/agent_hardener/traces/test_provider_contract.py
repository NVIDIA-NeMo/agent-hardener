# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The contract every TraceProvider must keep, run against each real backend.

A backend joins by adding a :class:`Harness` to ``HARNESSES``: it writes requests into the backend's
own source format, resets the source (the equivalent of a sandbox recreate), and breaks the raw copy.
Everything below then runs against it unchanged. A new provider only has to pass this suite plus its
own fixture mapping tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

import pytest

from agent_hardener.traces import AgentTrace, TraceProvider
from agent_hardener.traces.relay import RAW_FILENAME, RelayTraceProvider

if TYPE_CHECKING:
    from pathlib import Path

REFUSAL = "agent-hardener[exfil_guard]: refused send_email (safety score 0.93 >= 0.50)"
DENIAL = "requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://evil.example/x"


@dataclass
class Request:
    """A backend-neutral description of one victim request."""

    input: str
    output: str = "done"
    tools: list[tuple[str, str, bool]] = field(default_factory=list)  # (name, result, raised)
    in_flight: bool = False


class Harness(Protocol):
    def provider(self) -> TraceProvider: ...
    def write(self, requests: list[Request]) -> None: ...
    def reset(self) -> None: ...
    def break_raw_copy(self) -> None: ...
    def raw_saved(self) -> bool: ...
    def write_garbage_tail(self) -> None: ...


class RelayHarness:
    """Writes ATOF like the example victim, into a fake sandbox file that ``sync`` copies out."""

    def __init__(self, tmp_path: Path) -> None:
        self.sandbox = tmp_path / "sandbox" / RAW_FILENAME
        self.host = tmp_path / "host" / RAW_FILENAME
        self.raw_dir = tmp_path / "run" / "traces"
        self.clock = datetime(2026, 10, 7, 12, 0, tzinfo=UTC)
        self.counter = 0

    def provider(self) -> TraceProvider:
        return RelayTraceProvider(self.host, self._sync, self.raw_dir)

    def _sync(self) -> None:
        if self.sandbox.is_file():
            self.host.parent.mkdir(parents=True, exist_ok=True)
            self.host.write_text(self.sandbox.read_text(encoding="utf-8"), encoding="utf-8")

    def write(self, requests: list[Request]) -> None:
        self.sandbox.parent.mkdir(parents=True, exist_ok=True)
        with self.sandbox.open("a", encoding="utf-8") as handle:
            for request in requests:
                for record in self._records(request):
                    handle.write(json.dumps(record) + "\n")

    def reset(self) -> None:
        self.sandbox.unlink()

    def break_raw_copy(self) -> None:
        self.raw_dir.parent.mkdir(parents=True, exist_ok=True)
        self.raw_dir.write_text("not a directory", encoding="utf-8")

    def raw_saved(self) -> bool:
        return (self.raw_dir / RAW_FILENAME).is_file()

    def write_garbage_tail(self) -> None:
        with self.sandbox.open("a", encoding="utf-8") as handle:
            handle.write('{"atof_version":"0.1","kind":"sco')

    def _records(self, request: Request) -> list[dict[str, Any]]:
        root = self._uuid()
        records = [self._scope("start", "agent", "agent-hardener.request", root, parent="propagation-root")]
        llm = self._uuid()
        records.append(
            self._scope(
                "start",
                "llm",
                "model",
                llm,
                root,
                category_profile={"annotated_request": {"messages": [{"role": "user", "content": request.input}]}},
            )
        )
        records.append(self._scope("end", "llm", "model", llm, root, category_profile={"annotated_response": {}}))
        for name, result, raised in request.tools:
            tool = self._uuid()
            records.append(self._scope("start", "tool", name, tool, root, data={"q": 1}))
            status = "ERROR" if raised else "OK"
            records.append(
                self._scope("end", "tool", name, tool, root, data=result, metadata={"otel.status_code": status})
            )
        final = self._uuid()
        records.append(self._scope("start", "llm", "model", final, root))
        records.append(
            self._scope(
                "end",
                "llm",
                "model",
                final,
                root,
                category_profile={"annotated_response": {"message": request.output}},
            )
        )
        if not request.in_flight:
            records.append(self._scope("end", "agent", "agent-hardener.request", root, parent="propagation-root"))
        return records

    def _uuid(self) -> str:
        self.counter += 1
        return f"uuid-{self.counter:04d}"

    def _scope(self, phase: str, category: str, name: str, uuid: str, parent: str, **extra: Any) -> dict[str, Any]:
        self.clock += timedelta(milliseconds=10)
        return {
            "atof_version": "0.1",
            "kind": "scope",
            "scope_category": phase,
            "category": category,
            "name": name,
            "uuid": uuid,
            "parent_uuid": parent,
            "timestamp": self.clock.isoformat(),
            **extra,
        }


HARNESSES = {"relay": RelayHarness}


@pytest.fixture(params=sorted(HARNESSES))
def harness(request: pytest.FixtureRequest, tmp_path: Path) -> Harness:
    return HARNESSES[request.param](tmp_path)


def test_an_empty_or_missing_source_returns_nothing(harness: Harness) -> None:
    assert harness.provider().get_traces() == []


def test_partly_written_data_does_not_raise(harness: Harness) -> None:
    harness.write([Request("hello")])
    harness.write_garbage_tail()
    assert [trace.input for trace in harness.provider().get_traces()] == ["hello"]


def test_in_flight_requests_are_skipped(harness: Harness) -> None:
    harness.write([Request("done one"), Request("still running", in_flight=True)])
    assert [trace.input for trace in harness.provider().get_traces()] == ["done one"]


def test_each_call_returns_fresh_data(harness: Harness) -> None:
    provider = harness.provider()
    harness.write([Request("first")])
    assert len(provider.get_traces()) == 1
    harness.write([Request("second")])
    assert [trace.input for trace in provider.get_traces()] == ["first", "second"]


def test_raw_data_is_saved(harness: Harness) -> None:
    harness.write([Request("hello")])
    assert len(harness.provider().get_traces()) == 1
    assert harness.raw_saved()


def test_a_failing_raw_copy_still_returns_traces(harness: Harness) -> None:
    harness.break_raw_copy()
    harness.write([Request("hello")])
    assert [trace.input for trace in harness.provider().get_traces()] == ["hello"]


def test_history_survives_a_source_reset(harness: Harness) -> None:
    provider = harness.provider()
    harness.write([Request("before recreate")])
    assert len(provider.get_traces()) == 1
    harness.reset()
    harness.write([Request("after recreate")])
    assert [trace.input for trace in provider.get_traces()] == ["before recreate", "after recreate"]


def test_tool_outcomes_are_classified_with_the_shared_signals(harness: Harness) -> None:
    harness.write(
        [
            Request(
                "go",
                tools=[
                    ("read", "fine", False),
                    ("send_email", REFUSAL, False),
                    ("fetch", DENIAL, True),
                    ("x", "boom", True),
                ],
            )
        ]
    )
    (trace,) = harness.provider().get_traces()
    assert [(call.name, call.outcome, call.rail) for call in trace.tool_calls] == [
        ("read", "ok", None),
        ("send_email", "guardrail_refused", "exfil_guard"),
        ("fetch", "network_denied", None),
        ("x", "error", None),
    ]
    assert all(call.ok == (call.outcome == "ok") for call in trace.tool_calls)


def test_traces_round_trip_through_json(harness: Harness) -> None:
    harness.write([Request("hello", tools=[("read", "fine", False)])])
    for trace in harness.provider().get_traces():
        assert AgentTrace.model_validate_json(trace.model_dump_json()) == trace
