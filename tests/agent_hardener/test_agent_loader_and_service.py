# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
import time
import types
from typing import Any, cast

import pytest

from agent_hardener.models import (
    AgentConfig,
    AgentRunInput,
    AttackRecord,
    DefenderAnalysis,
    TargetInput,
    ValidatorReport,
)
from agent_hardener.runtime.agent_loader import (
    RunCallable,
    _build_call_arguments,
    default_stub_output,
    default_stub_run,
    invoke_run_callable,
    load_run_callable,
    normalize_agent_output,
)
from agent_hardener.runtime.run_context import RunContext


def _request() -> AgentRunInput:
    return AgentRunInput(round_id="20260502T000000Z-abcdef12", target=TargetInput(name="target"))


def test_default_stub_sleeps_and_returns_role_specific_success(monkeypatch: Any) -> None:
    sleeps: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    def fake_uniform(_start: float, _end: float) -> float:
        return 3.5

    monkeypatch.setattr("agent_hardener.runtime.agent_loader.random.uniform", fake_uniform)
    monkeypatch.setattr("agent_hardener.runtime.agent_loader.asyncio.sleep", fake_sleep)

    agent = AgentConfig(name="stub attacker", role="attacker")
    output = asyncio.run(default_stub_run(_request(), agent))

    assert sleeps == [3.5]
    assert isinstance(output, AttackRecord)
    assert output.ok is True
    assert output.records == [{"target": "target", "status": "stubbed"}]


def test_load_run_callable_without_reference_uses_default_stub() -> None:
    assert load_run_callable(None) is default_stub_run


def test_default_stub_output_supports_every_role() -> None:
    request = _request()

    defender = default_stub_output(request, AgentConfig(name="defender", role="defender", capabilities="stub"))
    victim = default_stub_output(request, AgentConfig(name="victim", role="victim"))
    validator = default_stub_output(
        request.model_copy(update={"validator_kind": "benign"}),
        AgentConfig(name="validator", role="validator"),
    )

    assert isinstance(defender, DefenderAnalysis)
    assert defender.policy_patches == [{"operation": "merge", "path": "/", "value": {"stub": True}}]
    assert victim.summary == "stub victim completed for target"
    assert isinstance(validator, ValidatorReport)
    assert validator.kind == "benign"


def test_external_implementation_loading_and_invocation() -> None:
    module = types.ModuleType("external_agent_for_test")

    def run(request: AgentRunInput, agent: AgentConfig) -> dict[str, Any]:
        return {"summary": f"{agent.name}:{request.target.name}", "extra": "captured"}

    async def alternate(request: AgentRunInput) -> dict[str, Any]:
        return {"summary": request.round_id}

    module.run = run  # type: ignore[attr-defined]
    module.alternate = alternate  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    try:
        agent = AgentConfig(name="external", role="defender", implementation=module.__name__, capabilities="stub")
        request = _request()

        loaded = load_run_callable(module.__name__)
        output = asyncio.run(invoke_run_callable(loaded, request, agent))
        normalized = normalize_agent_output(agent, request, output)

        assert normalized.summary == "external:target"
        assert normalized.metadata["raw_output"] == {"extra": "captured"}

        loaded_alternate = load_run_callable(f"{module.__name__}:alternate")
        alternate_output = asyncio.run(invoke_run_callable(loaded_alternate, request, agent))
        assert alternate_output == {"summary": request.round_id}
    finally:
        sys.modules.pop(module.__name__, None)


def test_invoke_run_callable_signature_variants() -> None:
    request = _request()
    agent = AgentConfig(name="external", role="victim")

    def with_kwargs(**kwargs: Any) -> dict[str, Any]:
        return {"summary": kwargs["agent"].name}

    def with_two_positional(first: AgentRunInput, second: AgentConfig) -> dict[str, Any]:
        return {"summary": f"{first.target.name}:{second.name}"}

    def with_one_positional(first: AgentRunInput) -> dict[str, Any]:
        return {"summary": first.round_id}

    def with_awaitable_result() -> Any:
        async def inner() -> dict[str, str]:
            return {"summary": "awaited"}

        return inner()

    assert asyncio.run(invoke_run_callable(with_kwargs, request, agent)) == {"summary": "external"}
    assert asyncio.run(invoke_run_callable(with_two_positional, request, agent)) == {"summary": "target:external"}
    assert asyncio.run(invoke_run_callable(with_one_positional, request, agent)) == {
        "summary": "20260502T000000Z-abcdef12",
    }
    assert asyncio.run(invoke_run_callable(cast("RunCallable", with_awaitable_result), request, agent)) == {
        "summary": "awaited"
    }
    assert _build_call_arguments(cast("RunCallable", time.time), request, agent) == ([request, agent], {})


def test_invoke_run_callable_injects_ctx_only_when_declared() -> None:
    request = _request()
    agent = AgentConfig(name="external", role="victim")
    ctx = RunContext()
    received: list[RunContext] = []

    def named(request: AgentRunInput, agent: AgentConfig, ctx: RunContext) -> dict[str, Any]:
        received.append(ctx)
        return {"summary": "named"}

    def kwargs_only(**kwargs: Any) -> dict[str, Any]:
        received.append(kwargs["ctx"])
        return {"summary": "kwargs"}

    def no_ctx(request: AgentRunInput, agent: AgentConfig) -> dict[str, Any]:
        return {"summary": "no_ctx"}

    asyncio.run(invoke_run_callable(named, request, agent, ctx))
    asyncio.run(invoke_run_callable(kwargs_only, request, agent, ctx))
    # A callable that does not declare ctx must be called without it (no unexpected-kwarg error).
    assert asyncio.run(invoke_run_callable(no_ctx, request, agent, ctx)) == {"summary": "no_ctx"}
    assert received == [ctx, ctx]


def test_normalize_agent_output_accepts_models_strings_and_validator_config_kind() -> None:
    request = _request()
    attacker = AgentConfig(name="attacker", role="attacker")
    victim = AgentConfig(name="victim", role="victim")
    validator = AgentConfig(name="validator", role="validator", config={"kind": "benign"})

    from_model = normalize_agent_output(
        attacker,
        request,
        AttackRecord(agent_id="raw", agent_name="raw", summary="from model"),
    )
    victim_output = normalize_agent_output(victim, request, "plain summary")
    validator_output = normalize_agent_output(validator, request, {"summary": "checked", "kind": "attack"})

    assert from_model.summary == "from model"
    assert victim_output.summary == "plain summary"
    assert isinstance(validator_output, ValidatorReport)
    assert validator_output.kind == "benign"


def test_load_run_callable_rejects_non_callable() -> None:
    module = types.ModuleType("bad_agent_for_test")
    module.run = "not callable"  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    try:
        with pytest.raises(TypeError, match="non-callable"):
            load_run_callable(module.__name__)
    finally:
        sys.modules.pop(module.__name__, None)
