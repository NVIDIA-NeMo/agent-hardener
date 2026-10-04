# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from agent_hardener.errors import VictimUnavailableError
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput
from agent_hardener.rate_limits import RateLimitError
from agent_hardener.runtime.adapters import HTTPAgentRunner, InProcessAgentRunner, RoutingAgentRunner, _run_url


def _request() -> AgentRunInput:
    return AgentRunInput(round_id="20260502T000000Z-abcdef12", target=TargetInput(name="target"))


def test_in_process_adapter_returns_failure_output_for_bad_import() -> None:
    agent = AgentConfig(name="bad", role="attacker", implementation="missing.module")

    output = asyncio.run(InProcessAgentRunner().run(agent, _request()))

    assert output.ok is False
    assert output.error is not None
    assert "missing" in output.error


def test_in_process_adapter_enforces_timeout(monkeypatch: Any) -> None:
    def slow_run(request: AgentRunInput, agent: AgentConfig) -> dict[str, str]:
        time.sleep(0.05)
        return {"summary": "too late"}

    monkeypatch.setattr("agent_hardener.runtime.adapters.load_run_callable", lambda _reference: slow_run)
    agent = AgentConfig(name="slow", role="attacker", timeout_seconds=0.001)

    output = asyncio.run(InProcessAgentRunner().run(agent, _request()))

    assert output.ok is False
    assert output.error == "TimeoutError"


def test_in_process_adapter_reraises_rate_limit_errors(monkeypatch: Any) -> None:
    def rate_limited_run(request: AgentRunInput, agent: AgentConfig) -> dict[str, str]:
        raise RuntimeError("resource exhausted: rate limit exceeded")

    monkeypatch.setattr("agent_hardener.runtime.adapters.load_run_callable", lambda _reference: rate_limited_run)
    agent = AgentConfig(name="limited", role="attacker")

    with pytest.raises(RateLimitError, match="agent limited rate limited request"):
        asyncio.run(InProcessAgentRunner().run(agent, _request()))


def test_in_process_adapter_reraises_victim_unavailable(monkeypatch: Any) -> None:
    # An unreachable victim is fatal: it must propagate (abort), not be demoted to a soft ok=False that
    # then replays validators against a dead endpoint and reports "0 blocked".
    def unreachable_run(request: AgentRunInput, agent: AgentConfig) -> dict[str, str]:
        raise VictimUnavailableError("victim is unreachable")

    monkeypatch.setattr("agent_hardener.runtime.adapters.load_run_callable", lambda _reference: unreachable_run)
    agent = AgentConfig(name="openshell-victim", role="victim")

    with pytest.raises(VictimUnavailableError, match="unreachable"):
        asyncio.run(InProcessAgentRunner().run(agent, _request()))


def test_routing_adapter_uses_service_url_to_choose_http() -> None:
    class FakeAdapter:
        async def run(self, agent: AgentConfig, request: AgentRunInput, ctx: Any = None) -> Any:
            return agent.name

    routing = RoutingAgentRunner(http_adapter=FakeAdapter(), in_process_adapter=FakeAdapter())

    http_result = asyncio.run(
        routing.run(AgentConfig(name="http", role="attacker", service_url="http://service"), _request()),
    )
    local_result = asyncio.run(routing.run(AgentConfig(name="local", role="attacker"), _request()))

    assert http_result == "http"
    assert local_result == "local"


def test_http_adapter_returns_failure_output_when_service_url_missing() -> None:
    agent = AgentConfig(name="http", role="attacker")

    output = asyncio.run(HTTPAgentRunner().run(agent, _request()))

    assert output.ok is False
    assert output.error is not None
    assert "service_url" in output.error


def test_http_adapter_normalizes_success_response(monkeypatch: Any) -> None:
    class FakeResponse:
        status_code = 200

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {"summary": "http ok", "records": [{"id": 1}]}

    class FakeClient:
        def __init__(self) -> None:
            pass

        async def post(self, url: str, json: dict[str, Any], timeout: float = 30.0) -> FakeResponse:
            assert url == "http://service/run"
            assert json["round_id"] == "20260502T000000Z-abcdef12"
            assert timeout == 30.0
            return FakeResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("agent_hardener.runtime.adapters.httpx.AsyncClient", FakeClient)
    agent = AgentConfig(name="http", role="attacker", service_url="http://service/")

    output = asyncio.run(HTTPAgentRunner().run(agent, _request()))

    assert output.ok is True
    assert output.summary == "http ok"


def test_http_adapter_reraises_rate_limit_response(monkeypatch: Any) -> None:
    class FakeResponse:
        status_code = 429

        def __init__(self) -> None:
            self.headers = {"retry-after": "10"}

        def raise_for_status(self) -> None:
            raise AssertionError("rate-limit responses should be handled before generic HTTP errors")

        def json(self) -> dict[str, Any]:
            return {"error": "quota exceeded"}

    class FakeClient:
        async def post(self, url: str, json: dict[str, Any], timeout: float = 30.0) -> FakeResponse:
            assert url == "http://service/run"
            return FakeResponse()

        async def aclose(self) -> None:
            pass

    monkeypatch.setattr("agent_hardener.runtime.adapters.httpx.AsyncClient", FakeClient)
    agent = AgentConfig(name="http", role="attacker", service_url="http://service/")

    with pytest.raises(RateLimitError) as exc_info:
        asyncio.run(HTTPAgentRunner().run(agent, _request()))

    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after == "10"
    assert "agent service http rate limited request" in str(exc_info.value)


def test_run_url_keeps_explicit_run_endpoint() -> None:
    agent = AgentConfig(name="http", role="attacker", service_url="http://service/run/")

    assert _run_url(agent) == "http://service/run"
