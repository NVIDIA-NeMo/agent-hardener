# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from agent_hardener.models import DefenderInput, DefenderOutput

from .graph import build_defender_graph
from .schemas import GraphState


def run(defender_input: DefenderInput) -> DefenderOutput:
    graph = build_defender_graph()

    initial_state = GraphState(
        original_input=defender_input,
        current_config=defender_input.context.get("current_config", ""),
        relay_plugins_path=defender_input.relay_plugins_path,
        safety_llm=defender_input.context.get("safety_llm"),
    )

    final_state = graph.invoke(initial_state)
    result: DefenderOutput = final_state["final_output"]

    if result is None:
        return DefenderOutput(ok=False, error_message="Graph completed without producing output.")

    print(
        f"[guardrails_defender] result: ok={result.ok} error={result.error_message} policy_yaml={bool(result.new_policy_yaml)}"
    )
    return result
