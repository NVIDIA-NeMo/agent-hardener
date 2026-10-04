# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the EventType catalog — the contract NeMo Studio mirrors."""

from __future__ import annotations

import json

from agent_hardener.events import EventCategory, EventType, category_of


def test_event_type_serializes_as_its_plain_string_value() -> None:
    # Consumers (events.jsonl, the HTTP sink, Studio) see the bare string, not "EventType.X".
    assert EventType.ROUND_STARTED == "round_started"
    assert json.dumps({"event": EventType.ATTACK_SUMMARY}) == '{"event": "attack_summary"}'


def test_every_member_carries_a_category() -> None:
    for event in EventType:
        assert isinstance(event.category, EventCategory)


def test_category_of_resolves_known_and_rejects_unknown() -> None:
    assert category_of("agent_failed") is EventCategory.AGENT
    assert category_of(EventType.SYNTH_PHASE) is EventCategory.SYNTH
    assert category_of("not_an_event") is None
