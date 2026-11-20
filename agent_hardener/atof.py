# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read a victim's NeMo Relay ATOF event log and recover what tools it called.

Used to prove the victim is instrumented (:mod:`agent_hardener.preflight.relay`): a Relay-connected
agent emits ATOF, so an empty stream means the run would grade an unguarded victim as hardened.

It deliberately does **not** attribute a tool call to the attack that caused it. That needs a shared
root scope per request, which no harness emits on its own — only a victim whose invoke opens one
(the Fabric adapters do; a BYO victim passes Relay's callback handler in its own serving loop).
Agent Hardener does not require either, so the attacked tool stays inferred from the attack prompt
(:func:`agent_hardener.agents.defenders.defenders_manager._parse_attacked_tool`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

# A record belongs to ATOF if it carries the envelope. `atof_version` alone is enough; the kind check
# keeps us from claiming unrelated JSONL a user happens to have pointed us at.
_ENVELOPE_KINDS = frozenset({"scope", "mark"})

# Emitted by the Fabric DeepAgents adapter once a fault has dirtied Relay's scope stack. Everything
# after it is nested under a stale scope, so attribution silently degrades rather than failing.
_QUARANTINE_MARKER = "scope stack dirty"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One completed tool invocation recovered from the victim's ATOF log."""

    name: str
    args: Any
    result: Any
    ok: bool
    uuid: str


def load_records(path: Path) -> list[dict[str, Any]]:
    """Parse an ``events.atof.jsonl`` file into a list. See :func:`iter_records`."""
    return list(iter_records(path))


def is_atof_record(record: dict[str, Any]) -> bool:
    """Whether ``record`` carries the ATOF envelope."""
    return "atof_version" in record or record.get("kind") in _ENVELOPE_KINDS


def has_quarantined_scopes(records: Iterable[dict[str, Any]]) -> bool:
    """Whether Relay reported a dirty scope stack, which makes the stream untrustworthy.

    Surfaced in the run report rather than swallowed: a quarantined run must not read as clean.
    """
    return any(_QUARANTINE_MARKER in json.dumps(record.get("metadata") or {}) for record in records)


def tool_calls(records: Iterable[dict[str, Any]]) -> list[ToolCall]:
    """Recover completed tool calls.

    A tool call is a ``scope``/``tool`` pair: the ``start`` record carries the arguments, the ``end``
    record the result. Calls still in flight when the log was read are skipped — a tool that never
    returned has no outcome to attribute.
    """
    records = list(records)
    starts = {
        record["uuid"]: record
        for record in records
        if record.get("kind") == "scope" and record.get("scope_category") == "start" and "uuid" in record
    }
    calls: list[ToolCall] = []
    for record in records:
        if record.get("kind") != "scope" or record.get("scope_category") != "end":
            continue
        if record.get("category") != "tool":
            continue
        start = starts.get(record.get("uuid"))
        if start is None:
            continue  # an end without its start: the log was truncated ahead of us
        calls.append(
            ToolCall(
                name=start.get("name") or "",
                args=start.get("data"),
                result=record.get("data"),
                ok=succeeded(record),
                uuid=start["uuid"],
            )
        )
    return calls


# Keys an OTEL status can arrive under. Real Relay streams use the dotted `otel.status_code`.
_STATUS_KEYS = ("status_code", "otel_status_code", "otel.status_code")


def succeeded(end_record: dict[str, Any]) -> bool:
    """Whether the scope ended without an error status.

    Relay records an OTEL status on the scope end event, and a raising tool lands there as an error.
    A guardrail refusal does not: the plugin returns its refusal as the tool result, so it ends OK and
    only the result text shows it.
    """
    metadata = end_record.get("metadata")
    if not isinstance(metadata, dict):
        return True
    status = next((metadata[key] for key in _STATUS_KEYS if isinstance(metadata.get(key), str)), None)
    if status is not None:
        return status.upper() not in {"ERROR", "STATUS_CODE_ERROR"}
    return "error" not in metadata


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream ATOF records from ``path``, skipping blanks, non-ATOF JSON and a torn final line.

    The victim may still be writing when we read, so a half-flushed record must not fail the run.
    """
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except ValueError:
                continue
            if isinstance(record, dict) and is_atof_record(record):
                yield record


#: A ReAct agent names its tool in plain text instead of a structured call.
_REACT_ACTION = re.compile(r"^\s*Action:\s*`?([A-Za-z_][\w.-]*)", re.MULTILINE)


def model_scope_ends(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """The end records of the victim's model calls."""
    return [
        record
        for record in records
        if record.get("kind") == "scope" and record.get("scope_category") == "end" and record.get("category") == "llm"
    ]


def requested_tools(records: Iterable[dict[str, Any]]) -> list[str]:
    """The tools the victim's model asked to call, one entry per request, from its recorded responses.

    Independent of whether the call then ran through Relay, which is the point: a request with no matching
    ``tool`` scope is a tool call Relay did not see. Reads structured ``tool_calls`` wherever the
    framework nests them, and a ReAct ``Action: <tool>`` line for agents that pick tools in text.
    """
    names: list[str] = []
    for record in model_scope_ends(records):
        # LangChain keeps one call under several keys (``tool_calls``, ``additional_kwargs``, ...), each
        # with the same id; keyed by id, it counts once.
        structured = dict(_structured_tool_calls(record.get("data")))
        names.extend(structured.values() or _react_tool_names(record.get("data")))
    return names


def _structured_tool_calls(value: Any) -> Iterator[tuple[str, str]]:
    """``(call id, tool name)`` for every structured tool call nested in ``value``."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "tool_calls" and isinstance(item, list):
                yield from (call for entry in item if (call := _call(entry)) is not None)
            else:
                yield from _structured_tool_calls(item)
    elif isinstance(value, list):
        for item in value:
            yield from _structured_tool_calls(item)


def _call(entry: Any) -> tuple[str, str] | None:
    if not isinstance(entry, dict):
        return None
    function = entry.get("function")
    name = function.get("name") if isinstance(function, dict) else entry.get("name")
    if not isinstance(name, str) or not name:
        return None
    return str(entry.get("id") or name), name


def _react_tool_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return _REACT_ACTION.findall(value)
    if isinstance(value, dict):
        return [name for item in value.values() for name in _react_tool_names(item)]
    if isinstance(value, list):
        return [name for item in value for name in _react_tool_names(item)]
    return []
