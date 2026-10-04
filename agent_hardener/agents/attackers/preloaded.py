# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load preloaded attack hits as attacker output.

An alternative source of :class:`AttackRecord`s: instead of running the live ``agent_breaker``
attacker, replay attack hits recorded in JSON/JSONL files (e.g. ``agent-hardener run --replay``). This is
the attacker domain — the orchestrator consumes the resulting records the same way regardless of source.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from agent_hardener.agents.attackers.hits import normalize_hit_record
from agent_hardener.ids import normalize_agent_name
from agent_hardener.models import AttackRecord, PreloadedAttackConfig

if TYPE_CHECKING:
    from pathlib import Path


def load_preloaded_attacks(configs: list[PreloadedAttackConfig]) -> list[AttackRecord]:
    """Load configured attack hit records from JSON/JSONL files."""
    return [record for config in configs for record in _load_preloaded_attack(config)]


def _load_preloaded_attack(config: PreloadedAttackConfig) -> list[AttackRecord]:
    records = _read_preloaded_attack_records(config.path)
    attacks: list[AttackRecord] = []
    base_name = normalize_agent_name(config.name)
    for record_index, record in enumerate(records):
        normalized = normalize_hit_record(record, record_index, config.source)
        attacks.append(
            AttackRecord(
                agent_id=f"preloaded-{base_name}-{record_index}",
                agent_name=config.name,
                summary=f"preloaded attack hit {record_index} from {config.path}",
                records=[normalized],
                metadata={
                    "preloaded": True,
                    "path": str(config.path),
                    **config.metadata,
                },
            )
        )
    return attacks


def _read_preloaded_attack_records(path: Path) -> list[Any]:
    if not path.exists():
        msg = f"preloaded attack file does not exist: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        records: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                msg = f"invalid JSONL in {path} at line {line_number}: {exc}"
                raise ValueError(msg) from exc
        return records

    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("records", "hits", "attacks"):
                value = payload.get(key)
                if isinstance(value, list):
                    return value
            return [payload]
        return [{"value": payload}]

    msg = f"preloaded attack file must be .jsonl or .json: {path}"
    raise ValueError(msg)
