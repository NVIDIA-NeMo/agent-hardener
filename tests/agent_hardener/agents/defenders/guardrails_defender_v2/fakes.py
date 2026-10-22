# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A scripted stand-in for the chat model the guardrails defender's nodes build via ``get_llm``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.runnables import Runnable, RunnableLambda

from agent_hardener.agents.defenders.guardrails_defender_v2.schemas import GraphState
from agent_hardener.models import DefenderInput

if TYPE_CHECKING:
    from collections.abc import Callable


def prompt_text(value: Any) -> str:
    """Flatten a PromptValue or a list of chat messages into the text the model would have seen."""
    if hasattr(value, "to_string"):
        return str(value.to_string())
    return "\n".join(str(message.content) for message in value)


class FakeLLM(RunnableLambda[Any, Any]):
    """Chat model whose reply is computed from the prompt text; every prompt it saw is kept in ``calls``."""

    def __init__(self, respond: Callable[[str], Any]) -> None:
        self.calls: list[str] = []
        self.temperatures: list[float] = []
        self.structured_schema: type | None = None
        self._respond = respond
        super().__init__(self._reply)

    def _reply(self, value: Any) -> Any:
        text = prompt_text(value)
        self.calls.append(text)
        return self._respond(text)

    def with_structured_output(self, schema: type) -> Runnable[Any, Any]:
        self.structured_schema = schema
        return RunnableLambda(self._reply)


def defender_input(**overrides: Any) -> DefenderInput:
    fields: dict[str, Any] = {
        "attack_prompt": "read /etc/shadow",
        "agent_response": "root:x:0:0",
        "attacked_tool": "bash",
        "benign_requests": ["list files", "print the date"],
    }
    fields.update(overrides)
    return DefenderInput(**fields)


def graph_state(input_overrides: dict[str, Any] | None = None, **state_fields: Any) -> GraphState:
    return GraphState(original_input=defender_input(**(input_overrides or {})), **state_fields)
