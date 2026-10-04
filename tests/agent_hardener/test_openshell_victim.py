# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The OpenShell victim aborts loudly when the sandbox/victim is unreachable.

A transport-level failure (connection refused, server disconnected, timeout) means the victim crashed —
the run must raise VictimUnavailableError, not return a soft ok=False that lets attacks replay against a
dead victim and report a meaningless "0 blocked".
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from agent_hardener.agents.victims import openshell_victim as ov
from agent_hardener.errors import VictimUnavailableError
from agent_hardener.models import AgentConfig, AgentRunInput, TargetInput, VictimResult

_URL = "http://victim.test/v1/chat/completions"


def _request() -> AgentRunInput:
    return AgentRunInput(round_id="r1", target=TargetInput(name="victim", base_url=_URL))


def _agent() -> AgentConfig:
    return AgentConfig(name="openshell-victim", role="victim")


class _FakeClient:
    def __init__(self, *, raises: Exception | None = None, response: httpx.Response | None = None) -> None:
        self._raises = raises
        self._response = response

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *_a: object) -> bool:
        return False

    async def post(self, _url: str, **_kw: Any) -> httpx.Response:
        if self._raises is not None:
            raise self._raises
        assert self._response is not None
        return self._response


def _patch_client(monkeypatch: pytest.MonkeyPatch, client: _FakeClient) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", lambda *_a, **_k: client)


def test_server_disconnect_raises_victim_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(
        monkeypatch, _FakeClient(raises=httpx.RemoteProtocolError("Server disconnected without sending a response."))
    )
    with pytest.raises(VictimUnavailableError, match="unreachable"):
        asyncio.run(ov.run(_request(), _agent()))


def test_connection_refused_raises_victim_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_client(monkeypatch, _FakeClient(raises=httpx.ConnectError("[Errno 61] Connection refused")))
    with pytest.raises(VictimUnavailableError):
        asyncio.run(ov.run(_request(), _agent()))


def test_normal_response_returns_victim_result(monkeypatch: pytest.MonkeyPatch) -> None:
    response = httpx.Response(
        200,
        json={"choices": [{"message": {"content": "done"}}]},
        request=httpx.Request("POST", _URL),
    )
    _patch_client(monkeypatch, _FakeClient(response=response))
    result = asyncio.run(ov.run(_request(), _agent()))
    assert isinstance(result, VictimResult)
    assert result.ok is True  # a real response is still a normal result, not an abort
