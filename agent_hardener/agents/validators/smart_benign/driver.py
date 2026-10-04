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

from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from .graph import build_synth_graph, quiet_warnings
from .state import SynthState

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig

    # (questions: [{gap, question, options}]) -> answers: [{gap, answer}]
    AnswerProvider = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


async def drive_synth(
    state: SynthState,
    *,
    answer_provider: AnswerProvider,
    lf_config: dict[str, Any] | None = None,
    thread_id: str = "synth",
    on_node: Callable[[str], None] | None = None,
) -> SynthState:
    """Run synth to completion, calling *answer_provider* at each interview interrupt.

    *on_node* (optional) is invoked with each node name as it completes, for progress reporting.
    """
    graph = build_synth_graph(checkpointer=MemorySaver())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
    if lf_config and lf_config.get("callbacks"):
        config["callbacks"] = lf_config["callbacks"]
    graph_input: Any = state
    with quiet_warnings():
        while True:
            interrupted: Any = None
            final_values: Any = None
            async for mode, chunk in graph.astream(graph_input, config, stream_mode=["updates", "values"]):
                if mode == "updates" and isinstance(chunk, dict):
                    if "__interrupt__" in chunk:
                        interrupted = chunk["__interrupt__"]
                    elif on_node:
                        for node_name in chunk:
                            on_node(node_name)
                elif mode == "values":  # full state snapshot after each step
                    final_values = chunk
            if interrupted is None:
                return SynthState.model_validate(final_values)
            answers = await answer_provider(interrupted[0].value["questions"])
            graph_input = Command(resume=answers)
