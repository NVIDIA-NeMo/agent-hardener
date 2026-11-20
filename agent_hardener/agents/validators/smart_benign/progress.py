# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Synth progress as ``synth_phase`` events, so a minutes-long synthesis never looks stuck.

Every path that runs the synth graph (the ``synth-benign`` CLI, the ``run`` pre-flight, the HTTP service
behind Studio) feeds its per-node updates to a :class:`PhaseTracker`. Each finished node becomes one
``synth_phase`` event saying what it produced and how long it took, plus what runs next. Rendering is the
subscriber's job: the console renderer prints a line and moves its spinner, and Studio lists a recon step.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from agent_hardener.events import EventType
from agent_hardener.loggers import emit_event

from .graph import _ingestion_sources, _route_after_gap_detector
from .state import SynthState

logger = logging.getLogger(__name__)

#: What each synth-DAG node did, keyed by the node names registered in ``graph.py``.
PHASE_LABELS: dict[str, str] = {
    "nl_parser": "parsed capabilities from the description",
    "github_analyzer": "analyzed the GitHub repository",
    "api_prober": "probed the victim endpoint for tools",
    "profile_synthesizer": "merged the victim capability profile",
    "gap_detector": "checked the profile for gaps",
    "interview_compose": "composed the interview questions",
    "interview_ask": "recorded the interview answers",
    "request_generator": "generated the benign request suite",
    "critic": "filtered and deduplicated requests",
    "profile_writer": "wrote profile.json + requests.csv",
}

# What a node is doing while it runs, reported as the ``next`` of the node before it.
_RUNNING: dict[str, str] = {
    "profile_synthesizer": "merging the victim capability profile",
    "gap_detector": "checking the profile for gaps",
    "interview_compose": "composing interview questions",
    "interview_ask": "waiting for interview answers",
    "request_generator": "generating the benign request suite",
    "critic": "filtering and deduplicating requests",
    "profile_writer": "writing profile.json + requests.csv",
}

# The node that always follows each one; the ingestion nodes and the gap detector are routed instead.
_FOLLOWS = {
    "profile_synthesizer": "gap_detector",
    "interview_compose": "interview_ask",
    "request_generator": "critic",
    "critic": "profile_writer",
}

# Ingestion node → its key in ``SynthState.partials``, and its short name while it runs alongside the others.
_PARTIAL_KEYS = {"nl_parser": "nl", "github_analyzer": "github", "api_prober": "api_probe"}
_SOURCES = {"nl_parser": "the description", "github_analyzer": "GitHub repo", "api_prober": "victim endpoint"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _merge(state: SynthState, update: dict[str, Any]) -> SynthState:
    """Apply one node's update the way the graph does: reducer fields accumulate, the rest overwrite."""
    merged: dict[str, Any] = {}
    for key, value in update.items():
        field = SynthState.model_fields[key]  # the graph rejects updates outside its state schema
        reducer = next((m for m in field.metadata if callable(m)), None)
        merged[key] = reducer(getattr(state, key), value) if reducer else value
    return state.model_copy(update=merged)


class PhaseTracker:
    """Turn ``drive_synth``'s per-node updates into ``synth_phase`` events.

    Keeps a mirror of the graph state so each event can say what the node produced (tools, gaps, requests
    the critic kept) and which node the graph routes to next. Payload::

        {"phase": "critic", "label": "filtered and deduplicated requests", "detail": "58 → 42 kept",
         "seconds": 41.2, "next": "writing profile.json + requests.csv"}

    ``detail`` is ``""`` when the node has nothing to count; ``seconds`` is ``None`` for the interview, whose
    time is mostly the operator's.
    """

    def __init__(self, state: SynthState) -> None:
        self._state = state
        self._ingesting = {s for s in _ingestion_sources(state) if s in _PARTIAL_KEYS}
        self._started = self._last = time.monotonic()

    def first_step(self) -> str:
        """What the graph does first, before any node has finished."""
        return self._pending_ingestion() or _RUNNING["profile_synthesizer"]

    def on_node(self, name: str, update: dict[str, Any]) -> None:
        """Emit the finished node's ``synth_phase`` event; nodes without a label are ignored."""
        if name not in PHASE_LABELS:
            return
        now = time.monotonic()
        # Ingestion nodes run in parallel from the start, so each one's time is measured from there.
        took = now - (self._started if name in self._ingesting else self._last)
        self._last = now
        before = self._state
        self._state = _merge(before, update)
        self._ingesting.discard(name)

        payload = {
            "phase": name,
            "label": PHASE_LABELS[name],
            "detail": self._detail(name, update, before),
            "seconds": None if name == "interview_ask" else round(took, 1),
            "next": self._next(name),
        }
        logger.info("recon: %s%s", payload["label"], f" — {payload['detail']}" if payload["detail"] else "")
        emit_event(EventType.SYNTH_PHASE, payload)

    def _detail(self, name: str, update: dict[str, Any], before: SynthState) -> str:
        parts: list[str] = []
        after = self._state
        if name in _PARTIAL_KEYS:
            parts.append(_plural(len((update.get("partials") or {}).get(_PARTIAL_KEYS[name], [])), "tool"))
            if name == "api_prober":
                probe = (update.get("source_details") or {}).get("api_probe") or {}
                if probe_errors := len(probe.get("probe_errors") or []):
                    parts.append(_plural(probe_errors, "probe error"))
        elif name == "profile_synthesizer" and after.profile is not None:
            parts.append(_plural(len(after.profile.tools), "tool"))
            parts.append(_plural(len(after.profile.personas), "persona"))
            parts.append(f"{len(after.profile.out_of_scope)} out-of-scope")
        elif name == "gap_detector":
            answered = {gap for gap, _q, _a in after.interviewer_answers if gap}
            unresolved = [g for g in after.gaps if g not in answered]
            parts.append(f"{_plural(len(unresolved), 'gap')} unresolved")
        elif name == "interview_compose":
            parts.append(_plural(len(after.composed_questions), "question"))
        elif name == "interview_ask" and before.composed_questions:
            parts.append(f"{len(update.get('interviewer_answers') or [])} of {len(before.composed_questions)} answered")
        elif name == "request_generator":
            parts.append(_plural(len(after.requests), "request"))
        elif name == "critic":
            parts.append(f"{len(before.requests)} → {len(after.requests)} kept")
        if errors := len(update.get("errors") or []):
            parts.append(_plural(errors, "error"))
        return ", ".join(parts)

    def _next(self, name: str) -> str:
        if name in _PARTIAL_KEYS:
            return self.first_step()
        if name == "gap_detector":
            return _RUNNING[_route_after_gap_detector(self._state)]
        if name == "interview_ask":
            return "re-merging the profile with the answers"
        return _RUNNING[_FOLLOWS[name]] if name in _FOLLOWS else "finishing"

    def _pending_ingestion(self) -> str:
        """The ingestion nodes still running, in pipeline order (empty once they have all finished)."""
        pending = [_SOURCES[n] for n in _PARTIAL_KEYS if n in self._ingesting]
        return f"reading {', '.join(pending)}" if pending else ""
