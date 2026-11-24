# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import httpx
import pytest

from agent_hardener.agents.chat_http import AsyncServerErrorRetryTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

_RATE_LIMITED_503 = httpx.Response(503, json={"error": "rate limit exceeded"})


class _Victim:
    """Answers with the given responses in order and records each request body it got."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return self._responses.pop(0)


def _post(victim: _Victim) -> tuple[httpx.Response, list[float]]:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    async def post() -> httpx.Response:
        transport = AsyncServerErrorRetryTransport(httpx.MockTransport(victim), delays=(1.0, 3.0), sleep=sleep)
        async with httpx.AsyncClient(transport=transport) as client:
            return await client.post("https://victim.test/v1/chat/completions", json={"q": 1})

    return asyncio.run(post()), slept


def test_a_brief_server_error_is_retried_with_the_same_request() -> None:
    victim = _Victim(httpx.Response(502), httpx.Response(200, json={"ok": True}))

    response, slept = _post(victim)

    assert response.status_code == 200
    assert slept == [1.0]
    assert victim.bodies == [b'{"q":1}'] * 2


def test_a_persistent_server_error_reaches_the_caller_after_the_last_retry() -> None:
    victim = _Victim(*(httpx.Response(500, json={"error": f"boom {n}"}) for n in range(3)))

    response, slept = _post(victim)

    assert response.status_code == 500
    assert response.json() == {"error": "boom 2"}
    assert slept == [1.0, 3.0]


@pytest.mark.parametrize(
    "response",
    [httpx.Response(400, json={"error": "blocked by guardrail"}), _RATE_LIMITED_503],
    ids=["client-error", "rate-limited-503"],
)
def test_responses_that_are_not_victim_failures_are_not_retried(response: httpx.Response) -> None:
    victim = _Victim(response)

    returned, slept = _post(victim)

    assert (returned.status_code, slept, len(victim.bodies)) == (response.status_code, [], 1)


class _BrokenBody(httpx.AsyncByteStream):
    """A 5xx body the connection drops halfway through."""

    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadError("connection reset")
        yield b""

    async def aclose(self) -> None:
        self.closed = True


def test_a_server_error_whose_body_fails_to_read_is_closed_and_not_retried() -> None:
    body = _BrokenBody()
    victim = _Victim(httpx.Response(500, stream=body))

    with pytest.raises(httpx.ReadError):
        _post(victim)

    assert body.closed
    assert len(victim.bodies) == 1
