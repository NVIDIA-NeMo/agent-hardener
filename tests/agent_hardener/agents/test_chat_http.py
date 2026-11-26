# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import functools
from typing import TYPE_CHECKING

import httpx
import pytest

from agent_hardener.agents import chat_http
from agent_hardener.rate_limits import AsyncRetryTransport

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    _Post = Callable[["_Victim"], tuple[httpx.Response, list[float]]]


class _Victim:
    """Answers with the given responses in order and records each request body it got."""

    def __init__(self, *responses: httpx.Response) -> None:
        self._responses = list(responses)
        self.bodies: list[bytes] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(request.content)
        return self._responses.pop(0)


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> _Post:
    """POST through :func:`chat_http.victim_client` to a mock victim, recording the waits instead of sleeping."""

    def _post(victim: _Victim) -> tuple[httpx.Response, list[float]]:
        slept: list[float] = []

        async def sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(chat_http, "AsyncRetryTransport", functools.partial(AsyncRetryTransport, sleep=sleep))

        async def send() -> httpx.Response:
            async with chat_http.victim_client(timeout=5, transport=httpx.MockTransport(victim)) as client:
                return await client.post("https://victim.test/v1/chat/completions", json={"q": 1})

        return asyncio.run(send()), slept

    return _post


@pytest.mark.parametrize(
    "limited",
    [
        httpx.Response(429, json={"error": "Too Many Requests"}),
        httpx.Response(503, json={"error": "rate limit exceeded"}),
    ],
    ids=["429", "rate-limited-503"],
)
def test_a_victim_rate_limit_is_waited_out_then_the_answer_returned(post: _Post, limited: httpx.Response) -> None:
    """A victim passing its model's 429 through used to fail replay validators at once."""
    victim = _Victim(limited, httpx.Response(200, json={"answer": "hi"}))

    response, slept = post(victim)

    assert (response.status_code, response.json()) == (200, {"answer": "hi"})
    assert len(slept) == 1
    assert victim.bodies == [b'{"q":1}'] * 2


def test_a_brief_server_error_is_retried_with_the_same_request(post: _Post) -> None:
    victim = _Victim(httpx.Response(502), httpx.Response(200, json={"ok": True}))

    response, slept = post(victim)

    assert response.status_code == 200
    assert slept == [1.0]
    assert victim.bodies == [b'{"q":1}'] * 2


def test_a_persistent_server_error_reaches_the_caller_after_the_last_retry(post: _Post) -> None:
    victim = _Victim(*(httpx.Response(500, json={"error": f"boom {n}"}) for n in range(3)))

    response, slept = post(victim)

    assert response.status_code == 500
    assert response.json() == {"error": "boom 2"}
    assert slept == [1.0, 3.0]


def test_a_client_error_is_not_retried(post: _Post) -> None:
    victim = _Victim(httpx.Response(400, json={"error": "blocked by guardrail"}))

    response, slept = post(victim)

    assert (response.status_code, slept, len(victim.bodies)) == (400, [], 1)


class _BrokenBody(httpx.AsyncByteStream):
    """A 5xx body the connection drops halfway through."""

    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raise httpx.ReadError("connection reset")
        yield b""

    async def aclose(self) -> None:
        self.closed = True


def test_a_server_error_whose_body_fails_to_read_is_closed_and_not_retried(post: _Post) -> None:
    body = _BrokenBody()
    victim = _Victim(httpx.Response(500, stream=body))

    with pytest.raises(httpx.ReadError):
        post(victim)

    assert body.closed
    assert len(victim.bodies) == 1
