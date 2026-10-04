# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical catalog of EventBus event types — the single source of truth for what agent-hardener emits.

Every ``emit``/``emit_event`` call names an :class:`EventType` member instead of a bare string, so the set of
event names can't drift. Consumers (the run recorder, the HTTP event sink, NeMo Studio's Hardening tab) mirror
this vocabulary; each member carries its :class:`EventCategory`, which a consumer uses to route the event to
the right view (feed, swarm graph, interview modal). ``EventType`` is a ``StrEnum``, so a member serializes as
its plain string value while ``.category`` stays available in-process.
"""

from __future__ import annotations

from enum import StrEnum


class EventCategory(StrEnum):
    """Coarse grouping a consumer uses to decide how to render an event."""

    LIFECYCLE = "lifecycle"  # run-level status + raw console output
    ROUND = "round"  # round / iteration progression + reports
    PHASE = "phase"  # stage phase boundaries
    DEPLOY = "deploy"  # victim bring-up / control
    ATTACK = "attack"  # attack stage
    DEFENSE = "defense"  # defense stage
    AGENT = "agent"  # per-agent activity (attackers, defenders, invocations)
    SYNTH = "synth"  # benign-suite synth + interview


class EventType(StrEnum):
    """Every event name the EventBus can carry, each tagged with its render :class:`EventCategory`.

    Members are declared as ``(name, category)``; ``__new__`` keeps the enum *value* the plain string so
    serialization is unchanged, while exposing ``.category`` for in-process consumers.
    """

    category: EventCategory

    def __new__(cls, value: str, category: EventCategory) -> EventType:
        member = str.__new__(cls, value)
        member._value_ = value
        member.category = category
        return member

    # Run lifecycle / console
    STATUS_STARTED = ("status_started", EventCategory.LIFECYCLE)
    STATUS_COMPLETED = ("status_completed", EventCategory.LIFECYCLE)
    OUTPUT = ("output", EventCategory.LIFECYCLE)

    # Round / iteration progression
    ROUND_STARTED = ("round_started", EventCategory.ROUND)
    ROUND_COMPLETED = ("round_completed", EventCategory.ROUND)
    ITERATION_STARTED = ("iteration_started", EventCategory.ROUND)
    ITERATION_COMPLETED = ("iteration_completed", EventCategory.ROUND)
    REPORT_WRITTEN = ("report_written", EventCategory.ROUND)

    # Stage phase boundaries
    PHASE_STARTED = ("phase_started", EventCategory.PHASE)
    PHASE_COMPLETED = ("phase_completed", EventCategory.PHASE)

    # Victim deploy / control
    VICTIM_CONTROL_STARTED = ("victim_control_started", EventCategory.DEPLOY)
    VICTIM_CONTROL_COMPLETED = ("victim_control_completed", EventCategory.DEPLOY)
    OPENSHELL_UPLOAD = ("openshell_upload", EventCategory.DEPLOY)
    RELAY_POLICY_UPLOAD = ("relay_policy_upload", EventCategory.DEPLOY)
    VICTIM_WARNING = ("victim_warning", EventCategory.DEPLOY)

    # Attack stage
    PRELOADED_ATTACKS_LOADED = ("preloaded_attacks_loaded", EventCategory.ATTACK)
    ATTACKERS_COMPLETED = ("attackers_completed", EventCategory.ATTACK)
    ATTACK_SUMMARY = ("attack_summary", EventCategory.ATTACK)
    ARTIFACT_WRITTEN = ("artifact_written", EventCategory.ATTACK)

    # Defense stage
    ATTACKER_SUMMARIES_PREPARED = ("attacker_summaries_prepared", EventCategory.DEFENSE)
    DEFENDER_SUMMARY = ("defender_summary", EventCategory.DEFENSE)
    POLICY_PATCHES_AGGREGATED = ("policy_patches_aggregated", EventCategory.DEFENSE)

    # Per-agent activity
    AGENT_STARTED = ("agent_started", EventCategory.AGENT)
    AGENT_PROGRESS = ("agent_progress", EventCategory.AGENT)
    AGENT_COMPLETED = ("agent_completed", EventCategory.AGENT)
    AGENT_FAILED = ("agent_failed", EventCategory.AGENT)
    AGENT_EXCHANGE = ("agent_exchange", EventCategory.AGENT)  # one prompt<->response transcript row
    LLM_CALL = ("llm_call", EventCategory.AGENT)  # one internal LLM request<->completion, attributed to an agent

    # Benign-suite synth + interview
    SYNTH_PHASE = ("synth_phase", EventCategory.SYNTH)
    INTERVIEW_STARTED = ("interview_started", EventCategory.SYNTH)
    INTERVIEW_COMPLETED = ("interview_completed", EventCategory.SYNTH)


_BY_VALUE: dict[str, EventType] = {event.value: event for event in EventType}


def category_of(event: str) -> EventCategory | None:
    """Return the render category for *event*, or ``None`` if it isn't a known :class:`EventType`."""
    member = _BY_VALUE.get(event)
    return member.category if member is not None else None
