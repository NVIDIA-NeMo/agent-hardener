# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The attack stage streams one AGENT_EXCHANGE per garak hit so the UI can show each agent's transcript."""

from __future__ import annotations

from typing import Any

from agent_hardener.events import EventType
from agent_hardener.models import AttackRecord
from agent_hardener.runtime.stages import attack


class _Recorder:
    def __init__(self) -> None:
        self.emitted: list[tuple[EventType, dict[str, Any]]] = []

    def emit(self, event: EventType, **kwargs: Any) -> None:
        self.emitted.append((event, kwargs))


def _exchanges(rec: _Recorder) -> list[dict[str, Any]]:
    return [kwargs for event, kwargs in rec.emitted if event == EventType.AGENT_EXCHANGE]


def test_emit_attack_exchanges_streams_one_per_hit_with_fallback_keys() -> None:
    rec = _Recorder()
    outputs = [
        AttackRecord(
            agent_id="direct-prompt-attacker-abc123",
            agent_name="Direct Prompt Attacker",
            ok=True,
            records=[
                {"prompt": "do X", "output": "refused", "detector": "jailbreak"},
                {"attack_prompt": "do Y", "victim_response": "ok", "probe": "dan"},
            ],
        )
    ]

    attack._emit_attack_exchanges(rec, outputs)

    rows = _exchanges(rec)
    assert len(rows) == 2
    assert (rows[0]["request"], rows[0]["response"], rows[0]["label"]) == ("do X", "refused", "jailbreak")
    assert rows[0]["agent_name"] == "Direct Prompt Attacker"
    assert rows[0]["agent_role"] == "attacker"
    # Second hit uses the attack_prompt / victim_response fallback keys.
    assert (rows[1]["request"], rows[1]["response"], rows[1]["label"]) == ("do Y", "ok", "dan")


def test_emit_attack_exchanges_skips_hits_without_prompt_or_response() -> None:
    rec = _Recorder()
    outputs = [AttackRecord(agent_id="a", agent_name="A", records=[{"score": 1.0}])]

    attack._emit_attack_exchanges(rec, outputs)

    assert _exchanges(rec) == []
