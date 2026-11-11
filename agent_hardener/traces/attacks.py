# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attack traces: which trace belongs to which attack hit, and the recording a replay loads back.

The recording (``round_<N>/attack_traces.json``) is a list of self-contained entries
``{prompt, response, trace}``. It is written with the same models the rest of the pipeline uses, so a
replay loads it with no backend involved, and a file exported by any backend in this format works.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from agent_hardener.models.base import AgentHardenerModel
from agent_hardener.models.traces import AgentTrace
from agent_hardener.traces.matching import find_attack_trace, normalize

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from agent_hardener.models import AttackRecord

ATTACK_TRACES_FILENAME = "attack_traces.json"


class RecordedAttackTrace(AgentHardenerModel):
    """One attack hit and the trace of the attempt that produced it."""

    prompt: str
    response: str
    trace: AgentTrace


_RECORDING = TypeAdapter(list[RecordedAttackTrace])


def attack_key(prompt: str) -> str:
    """The key attack traces are stored and looked up under: the normalized prompt."""
    return normalize(prompt)


def record_prompt(record: dict[str, Any]) -> str:
    """The prompt of an attack record: a plain string, or the first text turn of a garak hitlog prompt."""
    prompt = record.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        for turn in prompt.get("turns") or []:
            content = turn.get("content") if isinstance(turn, dict) else None
            text = content.get("text") if isinstance(content, dict) else None
            if isinstance(text, str) and text:
                return text
    return ""


def record_response(record: dict[str, Any]) -> str:
    """The victim's response in an attack record (live ``response`` or hitlog ``output``)."""
    value = record.get("response") or record.get("output")
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and isinstance(value.get("text"), str):
        return value["text"]
    return ""


def collect_attack_traces(attacks: Iterable[AttackRecord], traces: list[AgentTrace]) -> list[RecordedAttackTrace]:
    """Match every attack hit to its trace, keeping only the hits that have one.

    ``traces`` should already be windowed to the attack stage. Each trace is claimed once, so a prompt
    the attacker sent twice is matched to two different attempts.
    """
    claimed: set[str] = set()
    found: list[RecordedAttackTrace] = []
    for attack in attacks:
        for record in attack.records:
            prompt = record_prompt(record)
            if not prompt:
                continue
            response = record_response(record)
            trace = find_attack_trace(traces, prompt, response, claimed=claimed)
            if trace is not None:
                found.append(RecordedAttackTrace(prompt=prompt, response=response, trace=trace))
    return found


def by_prompt(recorded: Iterable[RecordedAttackTrace]) -> dict[str, AgentTrace]:
    """Index recorded traces by :func:`attack_key`; a repeated prompt keeps its first attempt."""
    indexed: dict[str, AgentTrace] = {}
    for entry in recorded:
        indexed.setdefault(attack_key(entry.prompt), entry.trace)
    return indexed


def dump_recording(recorded: list[RecordedAttackTrace]) -> list[dict[str, Any]]:
    """The recording as JSON-ready data."""
    return _RECORDING.dump_python(recorded, mode="json")


def load_recording(path: Path) -> list[RecordedAttackTrace]:
    """Load a recording. Raises ``ValueError`` (pydantic's ``ValidationError``) on an invalid file."""
    return _RECORDING.validate_json(path.read_bytes())
