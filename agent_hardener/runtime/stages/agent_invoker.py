# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one shared mechanic composed by the agent stages: invoking a single agent.

:class:`AgentInvoker` owns the timed, evented, logged invocation of one agent through the
:class:`~agent_hardener.runtime.adapters.AgentRunner`, plus the per-agent context every stage passes
down. The module-level helpers turn ``gather`` results into typed outputs and summarize them.
"""

from __future__ import annotations

from time import perf_counter
from typing import TYPE_CHECKING, Any

from agent_hardener.events import EventType
from agent_hardener.loggers import using_current_agent
from agent_hardener.runtime.agent_loader import failure_output
from agent_hardener.runtime.run_context import RunContext

if TYPE_CHECKING:
    import logging

    from agent_hardener.loggers import JsonlEventLogger
    from agent_hardener.models import AgentConfig, AgentRunInput, AgentRunOutput, SessionConfig
    from agent_hardener.runtime.adapters import AgentRunner


class AgentInvoker:
    """Run a single agent with lifecycle events, timing, and structured logging."""

    def __init__(self, config: SessionConfig, agent_adapter: AgentRunner) -> None:
        self.config = config
        self.agent_adapter = agent_adapter

    def request_context(self) -> dict[str, Any]:
        """Return per-agent context enriched with run-level concurrency settings.

        ``storage_root`` is the stable per-target artifact root (matching the
        ``agent-hardener synth-benign`` CLI) so the smart benign validator resolves the same
        ``benign_profiles/<target>/`` directory it was synthesized into pre-flight.
        """
        return {
            **self.config.context,
            "run_concurrency": self.config.run.concurrency.model_dump(),
            "storage_root": str(self.config.storage.root_dir),
        }

    async def run_agent(
        self,
        agent: AgentConfig,
        request: AgentRunInput,
        phase: str,
        mission_id: str | None,
        logger: logging.Logger,
        events: JsonlEventLogger,
    ) -> AgentRunOutput:
        payload = agent_event_payload(agent, request, phase, mission_id)
        logger.info(
            "agent start phase=%s role=%s name=%s id=%s iteration=%s validator_kind=%s",
            phase,
            agent.role,
            agent.name,
            agent.agent_id,
            request.iteration,
            request.validator_kind,
        )
        events.emit(EventType.AGENT_STARTED, **payload)
        started_at = perf_counter()
        ctx = RunContext(
            emit_progress=lambda msg, cur=None, tot=None: events.emit(
                EventType.AGENT_PROGRESS,
                agent_id=agent.agent_id,
                agent_name=agent.name,
                phase=phase,
                message=msg,
                current=cur,
                total=tot,
            )
        )
        agent_identity = {
            "agent_id": agent.agent_id,
            "agent_name": agent.name,
            "agent_role": agent.role,
            "validator_kind": request.validator_kind,
        }
        try:
            with using_current_agent(agent_identity):
                output = await self.agent_adapter.run(agent, request, ctx)
        except Exception as exc:
            duration_seconds = perf_counter() - started_at
            logger.exception(
                "agent stop phase=%s role=%s name=%s id=%s iteration=%s ok=False duration_seconds=%.3f error=%s",
                phase,
                agent.role,
                agent.name,
                agent.agent_id,
                request.iteration,
                duration_seconds,
                str(exc),
            )
            events.emit(
                EventType.AGENT_FAILED,
                **payload,
                duration_seconds=round(duration_seconds, 6),
                error=str(exc),
            )
            raise

        duration_seconds = perf_counter() - started_at
        logger.info(
            "agent stop phase=%s role=%s name=%s id=%s iteration=%s ok=%s duration_seconds=%.3f summary=%s",
            phase,
            agent.role,
            agent.name,
            agent.agent_id,
            request.iteration,
            output.ok,
            duration_seconds,
            short_summary(output.summary),
        )
        events.emit(
            EventType.AGENT_COMPLETED,
            **payload,
            ok=output.ok,
            duration_seconds=round(duration_seconds, 6),
            summary=output.summary,
            error=output.error,
        )
        return output


def agent_event_payload(
    agent: AgentConfig,
    request: AgentRunInput,
    phase: str,
    mission_id: str | None,
) -> dict[str, Any]:
    """Return common structured fields for agent lifecycle events."""
    return {
        "round_id": request.round_id,
        "mission_id": mission_id,
        "iteration": request.iteration,
        "phase": phase,
        "agent_id": agent.agent_id,
        "agent_name": agent.name,
        "agent_role": agent.role,
        "validator_kind": request.validator_kind,
    }


def collect_outputs(
    results: list[Any],
    agents: list[AgentConfig],
    request: AgentRunInput,
    expected_type: type,
    logger: logging.Logger,
) -> list[Any]:
    """Convert gather results (which may include exceptions) into typed outputs."""
    outputs: list[Any] = []
    for agent, result in zip(agents, results, strict=True):
        if isinstance(result, BaseException):
            outputs.append(failure_output(agent, request, result))
        elif isinstance(result, expected_type):
            outputs.append(result)
        else:
            logger.warning(
                "expected %s from %s agent, got %s", expected_type.__name__, agent.role, type(result).__name__
            )
            err = TypeError(f"{agent.role} agent returned {type(result).__name__}, expected {expected_type.__name__}")
            outputs.append(failure_output(agent, request, err))
    return outputs


def all_ok(outputs: list[Any]) -> bool:
    """Return whether every output has ok=true."""
    return all(bool(getattr(output, "ok", False)) for output in outputs)


def short_summary(summary: str, limit: int = 200) -> str:
    """Keep readable logs compact even when agents return long summaries."""
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3]}..."
