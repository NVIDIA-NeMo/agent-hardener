# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""External implementation loading and output normalization."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    AgentRunOutput,
    AttackRecord,
    DefenderAnalysis,
    ValidatorKind,
    ValidatorReport,
    VictimResult,
)

if TYPE_CHECKING:
    from agent_hardener.runtime.run_context import RunContext

logger = logging.getLogger(__name__)

RunCallable = Callable[..., AgentRunOutput | dict[str, Any] | Awaitable[AgentRunOutput | dict[str, Any]]]


def load_run_callable(reference: str | None) -> RunCallable:
    """Load an external run callable, or return the default stub."""
    if reference is None:
        return default_stub_run

    module_name, _, attribute = reference.partition(":")
    attribute = attribute or "run"
    logger.warning(
        "importing user-configured module %r (attribute %r) — ensure the source is trusted", module_name, attribute
    )
    module = importlib.import_module(module_name)
    run_callable = getattr(module, attribute)
    if not callable(run_callable):
        msg = f"{reference} resolved to a non-callable object"
        raise TypeError(msg)
    return cast("RunCallable", run_callable)


async def invoke_run_callable(
    run_callable: RunCallable,
    request: AgentRunInput,
    agent: AgentConfig,
    ctx: RunContext | None = None,
) -> Any:
    """Invoke a run callable with a flexible but standard-friendly signature."""
    args, kwargs = _build_call_arguments(run_callable, request, agent, ctx)
    if inspect.iscoroutinefunction(run_callable):
        return await run_callable(*args, **kwargs)

    result = await asyncio.to_thread(run_callable, *args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


async def default_stub_run(request: AgentRunInput, agent: AgentConfig) -> AgentRunOutput:
    """Sleep briefly and return a success-shaped output for the configured role."""
    delay = random.uniform(2.0, 7.0)  # noqa: S311 - smoke-test jitter, not security-sensitive.
    await asyncio.sleep(delay)
    return default_stub_output(request, agent)


def default_stub_output(request: AgentRunInput, agent: AgentConfig) -> AgentRunOutput:
    """Return deterministic success output for the default stub."""
    if agent.role == "attacker":
        return AttackRecord(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            summary=f"stub attacker completed against {request.target.name}",
            records=[{"target": request.target.name, "status": "stubbed"}],
        )
    if agent.role == "defender":
        return DefenderAnalysis(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            summary="stub defender proposed a no-op policy patch",
            policy_patches=[{"operation": "merge", "path": "/", "value": {"stub": True}}],
        )
    if agent.role == "victim":
        return VictimResult(
            agent_id=agent.agent_id,
            agent_name=agent.name,
            summary=f"stub victim completed for {request.target.name}",
            observations={"target": request.target.name, "status": "stubbed"},
        )
    return ValidatorReport(
        agent_id=agent.agent_id,
        agent_name=agent.name,
        kind=_validator_kind(request, agent),
        summary="stub validator accepted the session",
    )


def normalize_agent_output(agent: AgentConfig, request: AgentRunInput, output: Any) -> AgentRunOutput:
    """Coerce external output into the configured role's response schema."""
    payload = _coerce_payload(output)
    if agent.role == "attacker":
        return AttackRecord(**_model_payload(AttackRecord, agent, payload))
    if agent.role == "defender":
        return DefenderAnalysis(**_model_payload(DefenderAnalysis, agent, payload))
    if agent.role == "victim":
        return VictimResult(**_model_payload(VictimResult, agent, payload))
    return ValidatorReport(
        **_model_payload(
            ValidatorReport,
            agent,
            payload,
            {"kind": _validator_kind(request, agent)},
        ),
    )


def failure_output(agent: AgentConfig, request: AgentRunInput, exc: BaseException) -> AgentRunOutput:
    """Build a role-shaped failure output when wrapper execution fails."""
    error = str(exc) or exc.__class__.__name__
    return normalize_agent_output(
        agent,
        request,
        {
            "ok": False,
            "summary": f"{agent.role} execution failed",
            "error": error,
        },
    )


def _build_call_arguments(
    run_callable: RunCallable,
    request: AgentRunInput,
    agent: AgentConfig,
    ctx: RunContext | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    try:
        signature = inspect.signature(run_callable)
    except (TypeError, ValueError):
        return [request, agent], {}

    parameters = list(signature.parameters.values())
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
        kwargs: dict[str, Any] = {"request": request, "agent": agent}
        if ctx is not None:
            kwargs["ctx"] = ctx
        return [], kwargs

    kwargs = {}
    if "request" in signature.parameters:
        kwargs["request"] = request
    if "agent" in signature.parameters:
        kwargs["agent"] = agent
    if "ctx" in signature.parameters and ctx is not None:
        kwargs["ctx"] = ctx
    if kwargs:
        return [], kwargs

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
    ]
    if len(positional) >= 2:
        return [request, agent], {}
    if len(positional) == 1:
        return [request], {}
    return [], {}


def _coerce_payload(output: Any) -> dict[str, Any]:
    if isinstance(output, BaseModel):
        return output.model_dump(mode="json")
    if isinstance(output, dict):
        return dict(output)
    return {"summary": str(output)}


def _model_payload(
    model_type: type[BaseModel],
    agent: AgentConfig,
    payload: dict[str, Any],
    extra_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    fields = set(model_type.model_fields)
    metadata = dict(payload.get("metadata") or {})
    unknown = {
        key: value
        for key, value in payload.items()
        if key not in fields and key not in {"agent_id", "agent_name", "kind", "metadata"}
    }
    if unknown:
        metadata["raw_output"] = unknown

    model_payload = {key: value for key, value in payload.items() if key in fields}
    model_payload["agent_id"] = agent.agent_id
    model_payload["agent_name"] = agent.name
    if metadata:
        model_payload["metadata"] = metadata
    if extra_base:
        model_payload.update(extra_base)
    return model_payload


def _validator_kind(request: AgentRunInput, agent: AgentConfig) -> ValidatorKind:
    configured_kind = agent.config.get("kind")
    if configured_kind in {"attack", "benign"}:
        return cast("ValidatorKind", configured_kind)
    if request.validator_kind is not None:
        return request.validator_kind
    logger.warning("validator kind not configured for agent %r, defaulting to 'attack'", agent.name)
    return "attack"
