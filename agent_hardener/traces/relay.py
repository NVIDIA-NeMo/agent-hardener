# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Traces from the victim's NeMo Relay ATOF event log.

One :class:`AgentTrace` per victim request. A request is the top-level ``agent`` scope the victim
opens per invocation (``agent-hardener.request`` in the example victim); its LLM and tool scopes, at
any depth, become the trace's steps. A victim that opens no such scope gives no traces, which every
consumer reads as "no traces" and handles by falling back.

The raw file is saved under ``raw_dir`` on every call. While the sandbox lives the file is
append-only, so each copy is a superset of the last and overwriting it loses nothing. A sandbox
recreate starts a fresh, shorter file: the previous copy is then kept as ``events.<n>.atof.jsonl``
before the new one is written, and traces are read from every kept generation, so history and every
recorded ``trace_id`` survive the recreate.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from agent_hardener import atof
from agent_hardener.models.traces import AgentTrace, LlmCall, ToolCall
from agent_hardener.traces.provider import TraceProvider
from agent_hardener.traces.signals import LATCH_EVENT, classify_tool_call, is_blocked_response, text_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

logger = logging.getLogger(__name__)

RAW_FILENAME = "events.atof.jsonl"
_GENERATION = re.compile(r"^events\.(\d+)\.atof\.jsonl$")
_FRACTION = re.compile(r"(\.\d{6})\d+")
#: nemo-relay >= 0.11 records LangChain tool results in this wrapper: display ``content``, ``is_error``,
#: and the encoded ``result``. Older releases record the ``ToolMessage`` itself.
TOOL_RESULT_SCHEMA = "nemo.relay.tool_result@1"


class RelayTraceProvider(TraceProvider):
    """Reads the ATOF stream the runner copies out of the sandbox.

    Args:
        atof_path: Where ``sync`` leaves the victim's ``events.atof.jsonl`` on the host.
        sync: Fetches the sandbox file to ``atof_path`` (the runner's ``_sync_atof``), best-effort.
        raw_dir: ``<run_dir>/traces/``, where the raw generations are kept for this run.
    """

    def __init__(self, atof_path: Path, sync: Callable[[], None], raw_dir: Path) -> None:
        self._atof_path = atof_path
        self._sync = sync
        self._raw_dir = raw_dir
        self._warned_quarantine = False

    def get_traces(self) -> list[AgentTrace]:
        try:
            self._sync()
        except Exception:
            logger.warning("could not fetch the victim's ATOF stream", exc_info=True)
        saved = self._save_raw()
        sources = self._generations() if saved else [*self._generations()[:-1], self._atof_path]
        by_id: dict[str, AgentTrace] = {}
        for path in sources:  # oldest first, so a later generation's copy of a trace wins
            for trace in self._read(path):
                by_id[trace.trace_id] = trace
        return sorted(by_id.values(), key=lambda trace: trace.started_at)

    def _save_raw(self) -> bool:
        """Copy the fetched file into ``raw_dir``, rotating the old copy after a reset. Never raises."""
        try:
            if not self._atof_path.is_file():
                return True
            current = self._atof_path.read_text(encoding="utf-8")
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            latest = self._raw_dir / RAW_FILENAME
            if latest.is_file():
                previous = latest.read_text(encoding="utf-8")
                if not current.startswith(previous):
                    latest.rename(self._raw_dir / f"events.{self._next_generation()}.atof.jsonl")
            latest.write_text(current, encoding="utf-8")
        except Exception:
            logger.warning("could not save the raw ATOF stream to %s", self._raw_dir, exc_info=True)
            return False
        return True

    def _generations(self) -> list[Path]:
        """Every kept file, oldest first: the rotated generations, then the latest copy."""
        if not self._raw_dir.is_dir():
            return []
        rotated = sorted(
            (path for path in self._raw_dir.iterdir() if _generation_number(path) is not None),
            key=lambda path: _generation_number(path) or 0,
        )
        latest = self._raw_dir / RAW_FILENAME
        return [*rotated, latest] if latest.is_file() else rotated

    def _next_generation(self) -> int:
        return max((_generation_number(path) or 0 for path in self._generations()), default=0) + 1

    def _read(self, path: Path) -> list[AgentTrace]:
        try:
            records = atof.load_records(path)
        except OSError:
            logger.warning("could not read %s", path, exc_info=True)
            return []
        if not self._warned_quarantine and atof.has_quarantined_scopes(records):
            self._warned_quarantine = True
            logger.warning("Relay reported a dirty scope stack; traces may attribute calls to the wrong request")
        return build_traces(records)


def _generation_number(path: Path) -> int | None:
    match = _GENERATION.match(path.name)
    return int(match.group(1)) if match else None


def build_traces(records: Iterable[dict[str, Any]]) -> list[AgentTrace]:
    """One trace per completed top-level ``agent`` scope, oldest first."""
    records = list(records)
    starts: dict[str, dict[str, Any]] = {}
    ends: dict[str, dict[str, Any]] = {}
    marks: list[dict[str, Any]] = []
    for record in records:
        uuid = record.get("uuid")
        if record.get("kind") == "mark":
            marks.append(record)
        elif record.get("kind") == "scope" and isinstance(uuid, str):
            if record.get("scope_category") == "start":
                starts[uuid] = record
            elif record.get("scope_category") == "end":
                ends[uuid] = record

    def root_of(uuid: str | None) -> str | None:
        seen: set[str] = set()
        while uuid in starts and uuid not in seen:
            seen.add(uuid)
            parent = starts[uuid].get("parent_uuid")
            if parent not in starts:
                return uuid
            uuid = parent
        return None

    members: dict[str, list[str]] = {}
    for uuid in starts:
        root = root_of(uuid)
        if root is not None and root != uuid:
            members.setdefault(root, []).append(uuid)
    latched_roots = {root_of(mark.get("parent_uuid")) for mark in marks if LATCH_EVENT in text_of(mark.get("name"))}

    traces: list[AgentTrace] = []
    for root, start in starts.items():
        if start.get("category") != "agent" or start.get("parent_uuid") in starts or root not in ends:
            continue  # not a request, or still in flight
        traces.append(_trace(root, start, ends[root], members.get(root, []), starts, ends, root in latched_roots))
    return sorted(traces, key=lambda trace: trace.started_at)


def _trace(
    root: str,
    start: dict[str, Any],
    end: dict[str, Any],
    members: list[str],
    starts: dict[str, dict[str, Any]],
    ends: dict[str, dict[str, Any]],
    latched: bool,
) -> AgentTrace:
    steps: list[LlmCall | ToolCall] = []
    first_messages: list[dict[str, Any]] | None = None
    for uuid in sorted((u for u in members if u in ends), key=lambda u: starts[u].get("timestamp") or ""):
        step_start, step_end = starts[uuid], ends[uuid]
        category = step_start.get("category")
        if category == "llm":
            messages = _profile(step_start, "annotated_request").get("messages")
            messages = messages if isinstance(messages, list) else []
            if first_messages is None:
                first_messages = messages
            output = _profile(step_end, "annotated_response").get("message")
            latched = latched or is_blocked_response(_profile(step_end, "annotated_response") or step_end.get("data"))
            steps.append(LlmCall(messages=messages, output=output if isinstance(output, str) else ""))
        elif category == "tool":
            result, failed = _tool_result(step_end.get("data"))
            outcome, rail = classify_tool_call(result, atof.succeeded(step_end) and not failed)
            steps.append(
                ToolCall(
                    name=step_start.get("name") or "",
                    args=step_start.get("data"),
                    result=result,
                    ok=outcome == "ok",
                    outcome=outcome,
                    rail=rail,
                )
            )
    llm_outputs = [step.output for step in steps if isinstance(step, LlmCall) and step.output]
    return AgentTrace(
        trace_id=root,
        input=_text(start.get("data")) or _first_user_message(first_messages or []),
        output=_text(end.get("data")) or (llm_outputs[-1].strip() if llm_outputs else ""),
        started_at=_timestamp(start),
        ended_at=_timestamp(end),
        steps=steps,
        guardrail_latched=latched,
    )


def _profile(record: dict[str, Any], key: str) -> dict[str, Any]:
    profile = record.get("category_profile")
    value = profile.get(key) if isinstance(profile, dict) else None
    return value if isinstance(value, dict) else {}


def _tool_result(data: Any) -> tuple[Any, bool]:
    """The tool's output and whether it failed, from either recorded shape.

    nemo-relay >= 0.11 wraps the result (:data:`TOOL_RESULT_SCHEMA`): its ``content`` is what the agent
    saw, ``is_error`` the failure flag. Older releases record a LangChain ``ToolMessage`` directly.
    """
    if isinstance(data, dict) and data.get("schema") == TOOL_RESULT_SCHEMA:
        content = data.get("content")
        return (content if content is not None else data.get("result")), data.get("is_error") is True
    return data, _langchain_error(data)


def _langchain_error(result: Any) -> bool:
    """A LangChain ``ToolMessage`` marked ``status="error"``: the tool failed but the scope ended OK."""
    if not isinstance(result, dict):
        return False
    inner = result.get("data")
    return isinstance(inner, dict) and inner.get("type") == "tool" and inner.get("status") == "error"


def _first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "user":
            return _text(message.get("content"))
    return ""


def _text(value: Any) -> str:
    """Text from a plain string or an OpenAI-style list of content parts; ``""`` for anything else."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(part.get("text", "") for part in value if isinstance(part, dict))
    return ""


def _timestamp(record: dict[str, Any]) -> datetime:
    """ATOF timestamps carry nanoseconds; Python 3.11 parses at most microseconds."""
    raw = record.get("timestamp")
    if isinstance(raw, str):
        try:
            parsed = datetime.fromisoformat(_FRACTION.sub(r"\1", raw.replace("Z", "+00:00")))
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return datetime.min.replace(tzinfo=UTC)
