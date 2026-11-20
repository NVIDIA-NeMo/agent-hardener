# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drive the synth graph to completion, resuming it across interview ``interrupt()`` pauses.

The drive-to-completion loop for the *synchronous* front-ends (CLI + orchestrator pre-flight): run the
graph until it finishes or interrupts, hand the questions to an ``answer_provider`` (which decides the
transport — a terminal prompt), resume with the answers, repeat.

The HTTP service (``serve/app.py``) drives the *same graph* but does NOT use this loop: it is
request-driven — one ``graph.ainvoke`` per HTTP request advances to the next interrupt and returns, with
state persisted by the checkpointer across requests. Push (here) vs pull (serve) are different control
models; don't try to unify them.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from agent_hardener.events import EventType
from agent_hardener.loggers import emit_event

from .graph import build_synth_graph, quiet_warnings
from .state import SynthState
from .subgraphs import profile_writer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.graph.state import CompiledStateGraph

    # (questions: [{gap, question, options}]) -> answers: [{gap, answer}]
    AnswerProvider = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


async def drive_synth(
    state: SynthState,
    *,
    answer_provider: AnswerProvider,
    lf_config: dict[str, Any] | None = None,
    thread_id: str = "synth",
    on_node: Callable[[str, dict[str, Any]], None] | None = None,
) -> SynthState:
    """Run synth to completion, calling *answer_provider* at each interview interrupt.

    *on_node* (optional) is invoked with each node's name and its state update as it completes, for progress
    reporting.
    """
    graph = build_synth_graph(checkpointer=MemorySaver())
    usage = UsageMetadataCallbackHandler()
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [*((lf_config or {}).get("callbacks") or []), usage],
    }
    started = time.monotonic()
    graph_input: Any = state
    with quiet_warnings():
        while True:
            final_values, interrupted = await advance(graph, graph_input, config, on_node=on_node)
            if interrupted is None:
                final = SynthState.model_validate(final_values)
                if final.profile is not None:  # beside the artifacts the writer produced, never on its own
                    profile_writer.write_usage(
                        profile_writer.target_dir(final), dict(usage.usage_metadata), time.monotonic() - started
                    )
                return final
            # Bracket the prompt so a live display (the console spinner) releases the terminal while it runs.
            emit_event(EventType.INTERVIEW_STARTED, {})
            try:
                answers = await answer_provider(interrupted[0].value["questions"])
            finally:
                emit_event(EventType.INTERVIEW_COMPLETED, {})
            graph_input = Command(resume=answers)


async def advance(
    graph: CompiledStateGraph,
    graph_input: Any,
    config: RunnableConfig,
    *,
    on_node: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[Any, Any]:
    """Run *graph* until it finishes or interrupts, calling *on_node* as each node completes.

    Shared by :func:`drive_synth` and the HTTP service, so both report progress the same way. Returns the last
    state snapshot and the interrupts (``None`` when the graph finished).
    """
    interrupted: Any = None
    final_values: Any = None
    async for mode, chunk in graph.astream(graph_input, config, stream_mode=["updates", "values"]):
        if mode == "updates" and isinstance(chunk, dict):
            if "__interrupt__" in chunk:
                interrupted = chunk["__interrupt__"]
            elif on_node:
                for node_name, update in chunk.items():
                    on_node(node_name, update if isinstance(update, dict) else {})
        elif mode == "values":  # full state snapshot after each step
            final_values = chunk
    return final_values, interrupted
