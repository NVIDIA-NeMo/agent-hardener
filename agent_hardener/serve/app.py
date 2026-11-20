# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""FastAPI synth service: run the benign-suite synth graph with human-in-the-loop interview + review over HTTP.

One long-lived process holds the ``MemorySaver``-compiled graph, so a synth run can pause at each interview
``interrupt()`` and resume across separate requests (keyed by ``thread_id``). The nemo plugin (and any UI)
spawns ``agent-hardener serve`` and drives:

    POST /synth                     -> {thread_id, status: "interview"|"review", ...}
    POST /synth/{thread_id}/answers -> next interview round or the review suite
    POST /synth/{thread_id}/suite   -> write the reviewed requests.csv, status "done"

Localhost-only. Synth probes the live victim (`api_prober`), so the victim's ``base_url`` must be reachable.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import uvicorn
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

from agent_hardener.agents.validators.smart_benign.driver import advance
from agent_hardener.agents.validators.smart_benign.graph import build_synth_graph
from agent_hardener.agents.validators.smart_benign.models import GeneratedRequest
from agent_hardener.agents.validators.smart_benign.progress import PhaseTracker
from agent_hardener.agents.validators.smart_benign.state import SynthState
from agent_hardener.agents.validators.smart_benign.subgraphs.profile_writer.nodes.write import _write_requests_csv
from agent_hardener.agents.validators.smart_benign.validator import build_synth_inputs
from agent_hardener.agents.validators.smart_benign.wizard import _resolve_validator
from agent_hardener.config import load_config
from agent_hardener.loggers import using_event_sink
from agent_hardener.runtime.event_sink import make_http_event_sink
from agent_hardener.swarm_tracker import benign_profiles_dir


class SynthStart(BaseModel):
    """Body for ``POST /synth`` — the manifest/config path and an optional validator name."""

    config: str
    validator: str | None = None


class AnswersIn(BaseModel):
    """Body for ``POST /synth/{thread_id}/answers`` — one interview round's answers."""

    answers: list[dict[str, Any]] = []


class SuiteIn(BaseModel):
    """Body for ``POST /synth/{thread_id}/suite`` — the operator's reviewed benign suite."""

    suite: list[dict[str, Any]] = []


@dataclass
class _Session:
    """Per-run state the endpoints need beyond the checkpointer (which holds the graph state)."""

    artifact_dir: Path
    tracker: PhaseTracker  # turns the graph's node updates into synth_phase events across requests


def _serialize_suite(requests: list[GeneratedRequest]) -> list[dict[str, str]]:
    return [
        {"tool": r.tool, "payload": r.payload, "label": r.label, "rationale": r.rationale, "persona": r.persona or ""}
        for r in requests
    ]


def _deserialize_suite(rows: list[dict[str, Any]]) -> list[GeneratedRequest]:
    return [
        GeneratedRequest(
            tool=row["tool"],
            payload=row["payload"],
            label=row.get("label") or "benign",
            rationale=row.get("rationale") or "",
            persona=row.get("persona") or None,
        )
        for row in rows
        if row.get("tool") and row.get("payload")
    ]


def create_app() -> FastAPI:
    """Build the synth service. Holds one checkpointed graph + a thread_id -> session map for the process."""
    app = FastAPI(title="Agent Hardener Synth Service")
    graph = build_synth_graph(checkpointer=MemorySaver())
    sessions: dict[str, _Session] = {}
    # The NeMo plugin points this at the run Studio is watching, so each finished synth step shows up in its
    # recon checklist. Unset (a bare `agent-hardener serve`) → events go nowhere.
    event_sink = make_http_event_sink()

    async def _advance(thread_id: str, graph_input: Any) -> dict[str, Any]:
        """Run the graph to its next interrupt or the end, emitting a synth_phase event per finished node.

        Maps the outcome to an interview (interrupt) or review (completed) response.
        """
        config: Any = {"configurable": {"thread_id": thread_id}}
        tracker = sessions[thread_id].tracker
        with using_event_sink(event_sink) if event_sink else contextlib.nullcontext():
            values, interrupts = await advance(graph, graph_input, config, on_node=tracker.on_node)
        if interrupts:
            return {
                "thread_id": thread_id,
                "status": "interview",
                "questions": interrupts[0].value.get("questions", []),
            }
        final = SynthState.model_validate(values)
        return {"thread_id": thread_id, "status": "review", "suite": _serialize_suite(final.requests)}

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/synth")
    async def start(body: SynthStart) -> dict[str, Any]:
        session = load_config(body.config)
        try:
            validator_agent = _resolve_validator(session, body.validator)
        except SystemExit as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        inputs = build_synth_inputs(
            validator_agent.config,
            target_name=session.target.name,
            base_url=session.target.base_url,
            workflow_config=session.target.agent_relay_plugins,
        )
        artifact_dir = benign_profiles_dir(session.storage.root_dir, session.target.name)
        # interactive=True routes into the interview; the graph interrupts for answers (no TTY needed).
        state = SynthState.from_inputs(inputs, interactive=True, artifact_dir=artifact_dir)
        thread_id = uuid.uuid4().hex
        sessions[thread_id] = _Session(artifact_dir=artifact_dir, tracker=PhaseTracker(state))
        return await _advance(thread_id, state)

    @app.post("/synth/{thread_id}/answers")
    async def answers(thread_id: str, body: AnswersIn) -> dict[str, Any]:
        if thread_id not in sessions:
            raise HTTPException(status_code=404, detail=f"unknown synth thread {thread_id!r}")
        return await _advance(thread_id, Command(resume=body.answers))

    @app.post("/synth/{thread_id}/suite")
    async def suite(thread_id: str, body: SuiteIn) -> dict[str, Any]:
        session = sessions.get(thread_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"unknown synth thread {thread_id!r}")
        # Overwrite requests.csv with the reviewed suite; the returned `benign_csv` path is what a consumer
        # hands to `agent-hardener run --benign-suite <path>` to validate against this edited suite.
        path = _write_requests_csv(session.artifact_dir, _deserialize_suite(body.suite))
        return {"thread_id": thread_id, "status": "done", "benign_csv": str(path)}

    @app.get("/synth/{thread_id}")
    async def status(thread_id: str) -> dict[str, Any]:
        if thread_id not in sessions:
            raise HTTPException(status_code=404, detail=f"unknown synth thread {thread_id!r}")
        return {"thread_id": thread_id, "status": "active"}

    return app


def run_server(*, host: str = "127.0.0.1", port: int = 8100) -> None:
    """Serve the synth app with uvicorn (localhost). Backs ``agent-hardener serve``."""
    uvicorn.run(create_app(), host=host, port=port)
