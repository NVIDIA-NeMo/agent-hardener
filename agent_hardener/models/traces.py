# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the victim did while answering one request, independent of the backend that recorded it.

These cross the agent boundary inside contracts (an HTTP defender receives them as JSON), so they are
pydantic models and must round-trip through JSON exactly. Nothing here is backend-specific: a field
is added only when a consumer needs it, and every provider fills it the same way.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from agent_hardener.models.base import AgentHardenerModel

#: Why a tool call did (or did not) complete. ``ok`` is the only success; the rest say who stopped it.
ToolOutcome = Literal["ok", "error", "guardrail_refused", "network_denied"]


class LlmCall(AgentHardenerModel):
    """One model call inside the request: only its answer.

    The prompt it was sent (the whole conversation so far) is deliberately not kept. Nothing reads it,
    and it would put the victim's full conversation, unredacted, into every saved trace.
    """

    kind: Literal["llm_call"] = "llm_call"
    output: str = ""

    @model_validator(mode="before")
    @classmethod
    def _drop_recorded_messages(cls, data: Any) -> Any:
        """Traces recorded before ``messages`` was dropped still load (without it)."""
        if isinstance(data, dict) and "messages" in data:
            return {key: value for key, value in data.items() if key != "messages"}
        return data


class ToolCall(AgentHardenerModel):
    """One completed tool call inside the request."""

    kind: Literal["tool_call"] = "tool_call"
    name: str
    args: Any = None
    result: Any = None
    ok: bool
    outcome: ToolOutcome
    rail: str | None = None


TraceStep = Annotated[LlmCall | ToolCall, Field(discriminator="kind")]


class AgentTrace(AgentHardenerModel):
    """One victim request: the prompt that opened it, its answer, and the calls in between."""

    trace_id: str
    input: str
    output: str
    started_at: datetime
    ended_at: datetime
    steps: list[TraceStep] = Field(default_factory=list)
    guardrail_latched: bool = False

    @property
    def tool_calls(self) -> list[ToolCall]:
        """The tool calls, in execution order."""
        return [step for step in self.steps if isinstance(step, ToolCall)]

    @property
    def llm_calls(self) -> list[LlmCall]:
        """The model calls, in execution order."""
        return [step for step in self.steps if isinstance(step, LlmCall)]
