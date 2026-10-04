# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-LLM-call telemetry.

A LangChain callback that emits one :data:`~agent_hardener.events.EventType.LLM_CALL` event for every model
call, attributed to the currently-executing agent (:func:`agent_hardener.loggers.current_agent`). Attached once
in :func:`agent_hardener.llm.build_chat_model`, so every agent's internal LLM traffic is captured without each
agent wiring its own callback. Unattributed calls (no ambient agent) are skipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import BaseCallbackHandler

from agent_hardener.events import EventType
from agent_hardener.loggers import current_agent, emit_event

if TYPE_CHECKING:
    from uuid import UUID

    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import LLMResult

_MAX_CHARS = 2000


def _clip(text: str) -> str:
    text = text.strip()
    return text if len(text) <= _MAX_CHARS else text[:_MAX_CHARS] + "…"


def emit_agent_exchange(
    *, request: str, response: str, label: str, ok: bool, blocked: bool | None = None, phase: str = "validators"
) -> None:
    """Emit one AGENT_EXCHANGE attributed to the ambient agent (no-op if unattributed).

    ``blocked`` records whether the victim blocked/refused this request — set for validator checks so the UI
    can mark each prompt allowed vs blocked.
    """
    agent = current_agent()
    if agent is None:
        return
    payload: dict[str, Any] = {
        "agent_id": agent.get("agent_id"),
        "agent_name": agent.get("agent_name"),
        "agent_role": agent.get("agent_role"),
        "validator_kind": agent.get("validator_kind"),
        "phase": phase,
        "request": _clip(request),
        "response": _clip(response),
        "label": label,
        "ok": ok,
    }
    if blocked is not None:
        payload["blocked"] = blocked
    emit_event(EventType.AGENT_EXCHANGE, payload)


class LlmTelemetryCallback(BaseCallbackHandler):
    """Emit an LLM_CALL event per model call, attributed to the ambient agent (no-op if unattributed)."""

    def __init__(self) -> None:
        # Prompts captured on start, keyed by run_id, paired with the completion on end.
        self._requests: dict[str, str] = {}

    def on_chat_model_start(
        self, serialized: dict[str, Any], messages: list[list[BaseMessage]], *, run_id: UUID, **kwargs: Any
    ) -> None:
        batch = messages[-1] if messages else []
        self._requests[str(run_id)] = "\n".join(f"{m.type}: {m.content!s}" for m in batch)

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], *, run_id: UUID, **kwargs: Any) -> None:
        self._requests[str(run_id)] = "\n\n".join(prompts)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        agent = current_agent()
        request = self._requests.pop(str(run_id), "")
        if agent is None:
            return
        text = ""
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            text = getattr(gen, "text", "") or getattr(getattr(gen, "message", None), "content", "") or ""
        model = (response.llm_output or {}).get("model_name", "") if response.llm_output else ""
        self._emit(agent, request, str(text), label=str(model), ok=True)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        agent = current_agent()
        request = self._requests.pop(str(run_id), "")
        if agent is None:
            return
        self._emit(agent, request, str(error), label="error", ok=False)

    @staticmethod
    def _emit(agent: dict[str, Any], request: str, response: str, *, label: str, ok: bool) -> None:
        emit_event(
            EventType.LLM_CALL,
            {
                "agent_id": agent.get("agent_id"),
                "agent_name": agent.get("agent_name"),
                "agent_role": agent.get("agent_role"),
                "validator_kind": agent.get("validator_kind"),
                "request": _clip(request),
                "response": _clip(response),
                "label": label,
                "ok": ok,
            },
        )
